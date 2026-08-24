import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')

from bmtk.simulator.dpointnet.custom_ops import (
    build_csr_connectivity,
    fused_cuda_available,
    fused_spike_currents,
    reorder_csr_values,
)
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops
from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import (
    _csr_index_dtype,
)
from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
    _fused_cuda_dtype_error,
    _validate_fused_cuda_option,
)


INDICES = np.array([
    [0, 0],
    [1, 0],
    [1, 2],
    [0, 1],
], dtype=np.int64)
SYNAPSE_TYPES = np.array([0, 1, 0, 1], dtype=np.int64)


def _reference_currents(spikes, master_weights, basis):
    post_ids = tf.constant(INDICES[:, 0], tf.int32)
    pre_ids = tf.constant(INDICES[:, 1], tf.int32)
    edge_basis = tf.gather(basis, SYNAPSE_TYPES)
    compute_weights = tf.cast(master_weights, spikes.dtype)
    batch_outputs = []
    for batch in range(spikes.shape[0]):
        edge_values = (
            tf.gather(spikes[batch], pre_ids)[:, tf.newaxis]
            * compute_weights[:, tf.newaxis]
            * edge_basis
        )
        batch_outputs.append(
            tf.math.unsorted_segment_sum(edge_values, post_ids, 2)
        )
    return tf.reshape(tf.stack(batch_outputs), [-1, basis.shape[1]])


def _metadata_values(connectivity):
    return tf.raw_ops.ReadVariableOp(
        resource=connectivity['metadata_handle'],
        dtype=tf.dtypes.as_dtype(connectivity['index_dtype']),
    )


def test_build_csr_connectivity_groups_edges_by_source():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    metadata = _metadata_values(connectivity).numpy()

    np.testing.assert_array_equal(metadata[:4], [0, 1, 0, 1])
    np.testing.assert_array_equal(metadata[4:8], [0, 1, 1, 0])
    np.testing.assert_array_equal(metadata[8:12], [0, 2, 3, 4])
    np.testing.assert_array_equal(metadata[12:], [0, 1, 3, 2])
    assert connectivity['index_dtype'] == 'uint32'
    assert _metadata_values(connectivity).dtype == tf.uint32


def test_csr_index_dtype_retains_int64_beyond_uint32():
    uint32_max = np.iinfo(np.uint32).max

    assert _csr_index_dtype(uint32_max, 1, 1, 1) == tf.uint32
    assert _csr_index_dtype(uint32_max + 1, 1, 1, 1) == tf.int64


@pytest.mark.parametrize(
    ('indices', 'synapse_types', 'message'),
    [
        (np.array([[2, 0]]), np.array([0]), 'Postsynaptic indices'),
        (np.array([[0, 0]]), np.array([2]), 'Synapse type indices'),
    ],
)
def test_build_csr_connectivity_rejects_unsafe_indices(
        indices, synapse_types, message):
    with pytest.raises(ValueError, match=message):
        build_csr_connectivity(indices, synapse_types, 1, 2, 2)


@pytest.mark.parametrize(
    ('indices', 'synapse_types'),
    [
        (np.array([[0.0, 0.5]]), np.array([0])),
        (np.array([[0, 0]]), np.array([0.5])),
        (np.array([[0, 0]]), np.array([np.nan])),
    ],
)
def test_build_csr_connectivity_rejects_non_integer_metadata(
        indices, synapse_types):
    with pytest.raises(ValueError, match='finite integer values'):
        build_csr_connectivity(indices, synapse_types, 1, 1, 1)


@pytest.mark.parametrize(
    ('indices', 'synapse_types', 'n_source_neurons', 'n_synapse_types'),
    [
        (
            np.array([[0, np.iinfo(np.uint64).max]], dtype=np.uint64),
            np.array([0]),
            int(np.iinfo(np.uint64).max) + 1,
            1,
        ),
        (
            np.array([[0, 0]]),
            np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
            1,
            int(np.iinfo(np.uint64).max) + 1,
        ),
    ],
)
def test_build_csr_connectivity_rejects_metadata_outside_int64(
        indices, synapse_types, n_source_neurons, n_synapse_types):
    with pytest.raises(ValueError, match='int64 range'):
        build_csr_connectivity(
            indices,
            synapse_types,
            n_source_neurons,
            1,
            n_synapse_types,
        )


@pytest.mark.parametrize(
    'value',
    [
        'auto',
        np.str_('auto'),
        b'auto',
        np.bytes_('auto'),
        np.array('auto'),
        np.array(b'auto'),
    ],
)
def test_fused_cuda_option_accepts_string_scalars(value):
    assert _validate_fused_cuda_option(value) == 'auto'


@pytest.mark.parametrize('value', [0, 1, np.bool_(False), np.bool_(True)])
def test_fused_cuda_option_rejects_non_boolean_lookalikes(value):
    with pytest.raises(ValueError, match='use_fused_cuda'):
        _validate_fused_cuda_option(value)


def test_fused_cuda_dtype_error_describes_unsupported_policy():
    assert _fused_cuda_dtype_error(tf.float16, tf.float32) is None
    message = _fused_cuda_dtype_error(tf.bfloat16, tf.float32)
    assert 'compute_dtype=bfloat16' in message
    assert 'variable_dtype=float32' in message


def test_fused_availability_respects_visible_devices(monkeypatch):
    monkeypatch.setattr(tf.config, 'get_visible_devices', lambda device_type: [])

    assert not csr_spike_ops.fused_cuda_available()


def test_fused_availability_rejects_multiple_visible_gpus(monkeypatch):
    monkeypatch.setattr(
        tf.config, 'get_visible_devices', lambda device_type: [object(), object()]
    )

    assert not csr_spike_ops.fused_cuda_available()


def test_fused_availability_rejects_incompatible_gpu(monkeypatch):
    gpu = object()
    monkeypatch.setattr(
        tf.config, 'get_visible_devices', lambda device_type: [gpu]
    )
    monkeypatch.setattr(
        tf.config.experimental,
        'get_device_details',
        lambda device: {'compute_capability': (6, 0)},
    )

    assert not csr_spike_ops.fused_cuda_available()


def test_csr_connectivity_close_releases_resource():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    handle = connectivity['metadata_handle']

    connectivity.close()

    with pytest.raises((tf.errors.NotFoundError, tf.errors.FailedPreconditionError)):
        tf.raw_ops.ReadVariableOp(resource=handle, dtype=tf.uint32)


@pytest.mark.skipif(not fused_cuda_available(), reason='Fused CUDA op is unavailable.')
@pytest.mark.parametrize('dtype', [tf.float16, tf.float32])
@pytest.mark.parametrize(
    'spike_values',
    [
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
)
def test_fused_recurrent_currents_match_forward_and_gradients(dtype, spike_values):
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(
        tf.cast(master_weights, dtype), connectivity
    )
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=dtype)
    spikes = tf.Variable(spike_values, dtype=dtype)

    with tf.GradientTape() as fused_tape:
        fused = fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )
        fused_loss = tf.reduce_sum(fused)
    fused_gradients = fused_tape.gradient(
        fused_loss, [spikes, master_weights]
    )

    with tf.GradientTape() as reference_tape:
        reference = _reference_currents(spikes, master_weights, basis)
        reference_loss = tf.reduce_sum(reference)
    reference_gradients = reference_tape.gradient(
        reference_loss, [spikes, master_weights]
    )

    tolerance = 2e-3 if dtype == tf.float16 else 1e-6
    np.testing.assert_allclose(
        fused.numpy(), reference.numpy(), rtol=tolerance, atol=tolerance
    )
    for fused_gradient, reference_gradient in zip(
            fused_gradients, reference_gradients):
        np.testing.assert_allclose(
            fused_gradient.numpy(),
            reference_gradient.numpy(),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(not fused_cuda_available(), reason='Fused CUDA op is unavailable.')
def test_fused_input_currents_preserve_counts_and_only_differentiate_weights():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)
    spike_counts = tf.Variable(
        [[2.0, 0.0, 3.0], [0.0, 4.0, 0.0]], dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        currents = fused_spike_currents(
            spike_counts,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=False,
        )
        loss = tf.reduce_sum(currents)
    spike_gradient, weight_gradient = tape.gradient(
        loss, [spike_counts, master_weights]
    )

    with tf.GradientTape() as reference_tape:
        reference = _reference_currents(spike_counts, master_weights, basis)
        reference_loss = tf.reduce_sum(reference)
    reference_weight_gradient = reference_tape.gradient(
        reference_loss, master_weights
    )

    assert spike_gradient is None
    np.testing.assert_allclose(currents.numpy(), reference.numpy())
    np.testing.assert_allclose(
        weight_gradient.numpy(), reference_weight_gradient.numpy()
    )


@pytest.mark.skipif(not fused_cuda_available(), reason='Fused CUDA op is unavailable.')
def test_fused_currents_execute_inside_tf_function():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    @tf.function
    def run(spikes):
        return fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )

    result = run(tf.constant([[1.0, 0.0, 2.0]], tf.float32))
    reference = _reference_currents(
        tf.constant([[1.0, 0.0, 2.0]], tf.float32),
        master_weights,
        basis,
    )
    np.testing.assert_allclose(result.numpy(), reference.numpy())


@pytest.mark.skipif(not fused_cuda_available(), reason='Fused CUDA op is unavailable.')
def test_fused_currents_support_int64_connectivity():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    metadata = _metadata_values(connectivity).numpy().astype(np.int64)
    with tf.device('/GPU:0'):
        metadata_handle = csr_spike_ops._create_metadata_resource(
            metadata, tf.int64
        )
    connectivity = csr_spike_ops.CsrConnectivity({
        **connectivity,
        'metadata_handle': metadata_handle,
        'index_dtype': 'int64',
    })
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)
    spikes = tf.constant([[1.0, 0.0, 2.0]], dtype=tf.float32)

    result = fused_spike_currents(
        spikes,
        master_weights,
        csr_weights,
        connectivity,
        basis,
        n_post=2,
        compute_spike_gradient=True,
    )

    np.testing.assert_allclose(
        result.numpy(),
        _reference_currents(spikes, master_weights, basis).numpy(),
    )


@pytest.mark.skipif(not fused_cuda_available(), reason='Fused CUDA op is unavailable.')
def test_fused_currents_support_empty_connectivity_and_gradients():
    connectivity = build_csr_connectivity(
        np.empty((0, 2), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        3,
        2,
        2,
    )
    master_weights = tf.Variable([], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)
    spikes = tf.Variable([[1.0, 0.0, 2.0]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        currents = fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )
        loss = tf.reduce_sum(currents)
    spike_gradient, weight_gradient = tape.gradient(
        loss, [spikes, master_weights]
    )

    np.testing.assert_array_equal(currents.numpy(), np.zeros((2, 2)))
    np.testing.assert_array_equal(spike_gradient.numpy(), np.zeros((1, 3)))
    assert weight_gradient.shape == (0,)


@pytest.mark.skipif(not fused_cuda_available(), reason='Fused CUDA op is unavailable.')
def test_fused_currents_reject_mismatched_master_shape():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0], dtype=tf.float32)
    csr_weights = tf.ones([4], dtype=tf.float32)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    with pytest.raises(ValueError, match='master_weights'):
        fused_spike_currents(
            tf.ones([1, 3]),
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )


@pytest.mark.skipif(not fused_cuda_available(), reason='Fused CUDA op is unavailable.')
def test_fused_currents_reject_dynamic_mismatched_master_shape():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    csr_weights = tf.ones([4], dtype=tf.float32)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def run(master_weights):
        return fused_spike_currents(
            tf.ones([1, 3]),
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )

    with pytest.raises(tf.errors.InvalidArgumentError, match='master_weights'):
        run(tf.ones([1], dtype=tf.float32))
