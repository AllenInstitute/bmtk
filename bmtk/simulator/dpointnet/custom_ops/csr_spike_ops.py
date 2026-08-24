import os
import itertools
import weakref
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


_LIBRARY_PATH = Path(__file__).with_name('_csr_spike_ops.so')
_ARCHITECTURE_PATH = Path(__file__).with_name('_csr_spike_ops.archs')
_OPS = None
_LOAD_ERROR = None
_RESOURCE_COUNTER = itertools.count()
_RESOURCE_CONTAINER = 'bmtk_dpointnet_csr'


def _read_built_architectures():
    if not _ARCHITECTURE_PATH.exists():
        return (), None
    values = {}
    for line in _ARCHITECTURE_PATH.read_text().splitlines():
        key, separator, value = line.partition('=')
        if separator:
            values[key] = value
    sm_architectures = tuple(
        int(value) for value in values.get('sm', '').split()
    )
    ptx_architecture = values.get('ptx')
    return sm_architectures, (
        int(ptx_architecture) if ptx_architecture else None
    )


_SM_ARCHITECTURES, _PTX_ARCHITECTURE = _read_built_architectures()


def _gpu_compatibility_error():
    visible_gpus = tf.config.get_visible_devices('GPU')
    if len(visible_gpus) != 1:
        return f'fused CUDA requires exactly one visible GPU; found {len(visible_gpus)}'
    if not _SM_ARCHITECTURES and _PTX_ARCHITECTURE is None:
        return f'build architecture metadata is missing at {_ARCHITECTURE_PATH}'
    details = tf.config.experimental.get_device_details(visible_gpus[0])
    capability = details.get('compute_capability')
    if capability is None:
        return f'compute capability is unavailable for {visible_gpus[0].name}'
    architecture = int(capability[0]) * 10 + int(capability[1])
    if architecture in _SM_ARCHITECTURES:
        return None
    if _PTX_ARCHITECTURE is not None and architecture >= _PTX_ARCHITECTURE:
        return None
    return (
        f'GPU compute capability sm_{architecture} is incompatible with '
        f'sm targets {_SM_ARCHITECTURES} and compute_{_PTX_ARCHITECTURE} PTX'
    )


def _destroy_metadata_resource(handle):
    try:
        tf.raw_ops.DestroyResourceOp(
            resource=handle, ignore_lookup_error=True
        )
    except (tf.errors.OpError, RuntimeError):
        pass


class CsrConnectivity(Mapping):
    def __init__(self, values):
        self._values = values
        self._finalizer = weakref.finalize(
            self, _destroy_metadata_resource, values['metadata_handle']
        )

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def close(self):
        self._finalizer()


def _environment_flag(name):
    value = os.environ.get(name, '').strip().lower()
    return value in ('1', 'true', 'yes', 'on')


if not _environment_flag('BMTK_DPOINTNET_DISABLE_FUSED_CUDA'):
    if _LIBRARY_PATH.exists():
        try:
            _OPS = tf.load_op_library(str(_LIBRARY_PATH))
        except (tf.errors.NotFoundError, OSError) as exc:
            _LOAD_ERROR = exc
    else:
        _LOAD_ERROR = FileNotFoundError(
            f'Fused DPointNet CUDA library does not exist at {_LIBRARY_PATH}.'
        )


def cuda_op_status():
    if _environment_flag('BMTK_DPOINTNET_DISABLE_FUSED_CUDA'):
        return 'disabled by BMTK_DPOINTNET_DISABLE_FUSED_CUDA'
    if _OPS is not None:
        compatibility_error = _gpu_compatibility_error()
        if compatibility_error is not None:
            return f'loaded, but {compatibility_error}'
        return f'loaded from {_LIBRARY_PATH}'
    return str(_LOAD_ERROR)


def fused_cuda_available():
    return _OPS is not None and _gpu_compatibility_error() is None


def _csr_index_dtype(
        n_edges, n_source_neurons, n_target_neurons, n_synapse_types):
    maximum_value = max(
        int(n_edges),
        int(n_source_neurons) - 1,
        int(n_target_neurons) - 1,
        int(n_synapse_types) - 1,
    )
    return tf.uint32 if maximum_value <= np.iinfo(np.uint32).max else tf.int64


def _create_metadata_resource(metadata, index_dtype):
    resource_name = f'csr_{os.getpid()}_{next(_RESOURCE_COUNTER)}'
    metadata_handle = tf.raw_ops.VarHandleOp(
        dtype=index_dtype,
        shape=metadata.shape,
        container=_RESOURCE_CONTAINER,
        shared_name=resource_name,
    )
    tf.raw_ops.AssignVariableOp(
        resource=metadata_handle,
        value=tf.constant(metadata, dtype=index_dtype),
    )
    return metadata_handle


def build_csr_connectivity(
        indices,
        synapse_types,
        n_source_neurons,
        n_target_neurons,
        n_synapse_types):
    indices = np.asarray(indices)
    synapse_types = np.asarray(synapse_types)
    for dimension, name in (
            (n_source_neurons, 'n_source_neurons'),
            (n_target_neurons, 'n_target_neurons'),
            (n_synapse_types, 'n_synapse_types')):
        if not isinstance(dimension, (int, np.integer)) or dimension <= 0:
            raise ValueError(f'{name} must be a positive integer.')
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f'indices must have shape [n_edges, 2], got {indices.shape}.')
    if synapse_types.shape != (indices.shape[0],):
        raise ValueError(
            'synapse_types must contain one value per edge, got '
            f'{synapse_types.shape} for {indices.shape[0]} edges.'
        )
    for values, name in (
            (indices, 'indices'),
            (synapse_types, 'synapse_types')):
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f'{name} must contain numeric integer values.')
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f'{name} must contain finite integer values.')
        int64_info = np.iinfo(np.int64)
        if np.any(values < int64_info.min) or np.any(values > int64_info.max):
            raise ValueError(f'{name} values must be within the int64 range.')

    indices = indices.astype(np.int64, copy=False)
    synapse_types = synapse_types.astype(np.int64, copy=False)
    pre_ids = indices[:, 1]
    if np.any(pre_ids < 0) or np.any(pre_ids >= n_source_neurons):
        raise ValueError('Presynaptic indices are outside the declared source dimension.')
    post_ids = indices[:, 0]
    if np.any(post_ids < 0) or np.any(post_ids >= n_target_neurons):
        raise ValueError('Postsynaptic indices are outside the declared target dimension.')
    if np.any(synapse_types < 0) or np.any(synapse_types >= n_synapse_types):
        raise ValueError('Synapse type indices are outside the basis table.')
    index_dtype = _csr_index_dtype(
        indices.shape[0],
        n_source_neurons,
        n_target_neurons,
        n_synapse_types,
    )
    numpy_index_dtype = index_dtype.as_numpy_dtype
    edge_ids = np.argsort(pre_ids, kind='stable').astype(
        numpy_index_dtype, copy=False
    )
    sorted_pre_ids = pre_ids[edge_ids]
    counts = np.bincount(sorted_pre_ids, minlength=n_source_neurons)
    row_splits = np.empty(
        n_source_neurons + 1, dtype=numpy_index_dtype
    )
    row_splits[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_splits[1:])

    device = '/GPU:0' if tf.config.get_visible_devices('GPU') else '/CPU:0'
    with tf.device(device):
        post_ids = indices[edge_ids, 0].astype(
            numpy_index_dtype, copy=False
        )
        sorted_synapse_types = synapse_types[edge_ids].astype(
            numpy_index_dtype, copy=False
        )
        metadata = np.concatenate(
            (post_ids, sorted_synapse_types, row_splits, edge_ids)
        )
        metadata_handle = _create_metadata_resource(
            metadata, index_dtype
        )
        return CsrConnectivity({
            'metadata_handle': metadata_handle,
            'index_dtype': index_dtype.name,
            'n_edges': int(indices.shape[0]),
            'n_sources': int(n_source_neurons),
            'n_post': int(n_target_neurons),
            'n_synapse_types': int(n_synapse_types),
        })


def reorder_csr_values(values, connectivity):
    if _OPS is None:
        raise RuntimeError(f'Fused DPointNet CUDA operator is unavailable: {cuda_op_status()}')
    return _OPS.dpointnet_csr_reorder(
        values,
        connectivity['metadata_handle'],
        Tindex=tf.dtypes.as_dtype(connectivity['index_dtype']),
        n_edges=connectivity['n_edges'],
    )


@ops.RegisterGradient('DpointnetCsrSpikeForward')
def _fused_spike_currents_gradient(op, current_grad):
    compute_spike_gradient = op.get_attr('compute_spike_gradient')
    n_post = op.get_attr('n_post')
    index_dtype = op.get_attr('Tindex')
    if compute_spike_gradient:
        spike_grad, weight_grad = _OPS.dpointnet_csr_spike_grad(
            op.inputs[0],
            current_grad,
            op.inputs[2],
            op.inputs[3],
            op.inputs[4],
            Tindex=index_dtype,
            n_post=n_post,
            n_edges=op.get_attr('n_edges'),
        )
    else:
        spike_grad = None
        weight_grad = _OPS.dpointnet_csr_weight_grad(
            op.inputs[0],
            current_grad,
            op.inputs[2],
            op.inputs[4],
            Tindex=index_dtype,
            n_post=n_post,
            n_edges=op.get_attr('n_edges'),
        )
    return (
        spike_grad,
        tf.cast(weight_grad, op.inputs[1].dtype),
        None,
        None,
        None,
    )


def fused_spike_currents(
        spikes,
        master_weights,
        csr_weights,
        connectivity,
        basis,
        n_post,
        compute_spike_gradient):
    if _OPS is None:
        raise RuntimeError(f'Fused DPointNet CUDA operator is unavailable: {cuda_op_status()}')
    if spikes.dtype not in (tf.float16, tf.float32):
        raise TypeError(
            f'Fused DPointNet CUDA operator requires float16 or float32 spikes, got {spikes.dtype}.'
        )
    if csr_weights.dtype != spikes.dtype or basis.dtype != spikes.dtype:
        raise TypeError('spikes, csr_weights, and basis must have the same dtype.')
    if master_weights.shape.rank != 1 or csr_weights.shape.rank != 1:
        raise ValueError('master_weights and csr_weights must be rank 1.')
    n_edges = connectivity['n_edges']
    if (master_weights.shape[0] is not None
            and master_weights.shape[0] != n_edges):
        raise ValueError(
            f'master_weights has {master_weights.shape[0]} values; '
            f'connectivity has {n_edges} edges.'
        )
    if csr_weights.shape[0] is not None and csr_weights.shape[0] != n_edges:
        raise ValueError(
            f'csr_weights has {csr_weights.shape[0]} values; '
            f'connectivity has {n_edges} edges.'
        )

    if connectivity['n_post'] != n_post:
        raise ValueError(
            f'Connectivity targets {connectivity["n_post"]} neurons, got n_post={n_post}.'
        )
    if connectivity['n_synapse_types'] != basis.shape[0]:
        raise ValueError(
            'Connectivity synapse types do not match the basis table: '
            f'{connectivity["n_synapse_types"]} != {basis.shape[0]}.'
        )

    return _OPS.dpointnet_csr_spike_forward(
        spikes,
        master_weights,
        connectivity['metadata_handle'],
        csr_weights,
        basis,
        Tindex=tf.dtypes.as_dtype(connectivity['index_dtype']),
        n_post=n_post,
        n_edges=connectivity['n_edges'],
        compute_spike_gradient=compute_spike_gradient,
    )
