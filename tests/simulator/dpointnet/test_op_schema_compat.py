"""Serialized develop graphs keep importing, executing and differentiating.

``DpointnetCsrSpikeForward`` keeps develop's schema; CSR-ordered FP32 masters use
the separately named ``DpointnetCsrSpikeForwardCsrMaster`` proxy.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, types_pb2

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops


# Develop (15d18e11) registration of DpointnetCsrSpikeForward.
DEVELOP_INPUTS = (
    ("spikes", "T"), ("master_weights", "Tmaster"), ("metadata", None), ("weights", "T"),
    ("basis", "T"), ("spike_gradient_scale", "T"), ("active_rows", None),
    ("incoming_pre_ids", "Tindex"), ("incoming_edge_ids", "Tindex"),
    ("incoming_types", "Tindex"), ("initial", "T"),
)
DEVELOP_BOOL_DEFAULTS = {
    "compute_weight_gradient": True, "use_grouped_batch32_forward": False,
    "use_fixed4_forward": False, "use_forward_run_aggregation": False,
    "use_device_active_queue_forward": False, "use_packed_sm120_backward": False,
    "write_csr_weight_gradient": False, "use_small_batch_backward": False,
    "use_javier_batch32_backward": False, "vjp_only": False,
}
DEVELOP_ATTRS = {
    "T", "Tmaster", "Tindex", "n_post", "n_edges", "n_pairs", "compute_spike_gradient",
    *DEVELOP_BOOL_DEFAULTS,
}


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


@pytest.fixture
def library():
    if ops._OPS is None:
        pytest.skip(f"DPointNet operator library unavailable: {ops.cuda_op_status()}")


def op_def(name):
    from tensorflow.python.framework import op_def_registry
    definition = op_def_registry.get(name)
    assert definition is not None, name
    return definition


def test_develop_forward_schema_is_unchanged(library):
    forward = op_def("DpointnetCsrSpikeForward")
    assert [(arg.name, arg.type_attr or None) for arg in forward.input_arg] == list(DEVELOP_INPUTS)
    assert {attr.name for attr in forward.attr} == DEVELOP_ATTRS
    for attr in forward.attr:
        if attr.name in DEVELOP_BOOL_DEFAULTS:
            assert attr.default_value.b is DEVELOP_BOOL_DEFAULTS[attr.name], attr.name
        elif attr.name == "compute_spike_gradient":
            assert not attr.HasField("default_value")
    master = op_def(ops.CSR_MASTER_FORWARD_OP)
    weights = [arg for arg in master.input_arg if arg.name == "weights"][0]
    assert weights.type == types_pb2.DT_FLOAT and not weights.type_attr
    assert {attr.name for attr in master.attr} == DEVELOP_ATTRS | {"uint8_types"}


def _placeholder(graph, name, dtype):
    node = graph.node.add(name=name, op="Placeholder")
    node.attr["dtype"].type = dtype
    return name


def develop_graph_def(dtype, index_dtype, n_post, n_edges, n_pairs, compute_spike_gradient=True):
    """A GraphDef node exactly as develop's Python wrapper serialized it (no W attribute)."""
    graph = graph_pb2.GraphDef()
    compute = tf.as_dtype(dtype).as_datatype_enum
    index = tf.as_dtype(index_dtype).as_datatype_enum
    inputs = [
        _placeholder(graph, "spikes", compute),
        _placeholder(graph, "master_weights", types_pb2.DT_FLOAT),
        _placeholder(graph, "metadata", types_pb2.DT_RESOURCE),
        _placeholder(graph, "weights", compute),
        _placeholder(graph, "basis", compute),
        _placeholder(graph, "spike_gradient_scale", compute),
        _placeholder(graph, "active_rows", types_pb2.DT_INT64),
        _placeholder(graph, "incoming_pre_ids", index),
        _placeholder(graph, "incoming_edge_ids", index),
        _placeholder(graph, "incoming_types", index),
        _placeholder(graph, "initial", compute),
    ]
    node = graph.node.add(name="forward", op="DpointnetCsrSpikeForward", input=inputs)
    attrs = {
        "T": attr_value_pb2.AttrValue(type=compute),
        "Tmaster": attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT),
        "Tindex": attr_value_pb2.AttrValue(type=index),
        "n_post": attr_value_pb2.AttrValue(i=n_post),
        "n_edges": attr_value_pb2.AttrValue(i=n_edges),
        "n_pairs": attr_value_pb2.AttrValue(i=n_pairs),
        "compute_spike_gradient": attr_value_pb2.AttrValue(b=compute_spike_gradient),
        **{name: attr_value_pb2.AttrValue(b=value) for name, value in DEVELOP_BOOL_DEFAULTS.items()},
    }
    for name, value in attrs.items():
        node.attr[name].CopyFrom(value)
    assert set(node.attr) == DEVELOP_ATTRS
    return graph.SerializeToString()


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("index_dtype", [tf.uint32, tf.int64])
def test_develop_graph_def_imports_with_the_rebuilt_library(library, dtype, index_dtype):
    graph_def = graph_pb2.GraphDef.FromString(develop_graph_def(dtype, index_dtype, 5, 39, 0))
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
    forward = graph.get_operation_by_name("forward")
    assert forward.type == "DpointnetCsrSpikeForward"
    assert forward.get_attr("T") == dtype
    assert forward.outputs[0].dtype == dtype
    for name in ("W", "csr_master_weights", "uint8_types"):
        with pytest.raises(ValueError):
            forward.get_attr(name)


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("index_dtype", [tf.uint32, tf.int64])
def test_develop_graph_def_executes_and_differentiates_on_gpu(library, dtype, index_dtype, monkeypatch):
    if not ops.fused_cuda_available():
        pytest.skip("Executing the native forward requires a supported GPU and rebuilt operators")
    from test_master_projection import oracle
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: index_dtype)
    rng = np.random.default_rng(485)
    width, posts, edges, batch = 7, 5, 39, 3
    indices = np.column_stack((rng.integers(posts, size=edges), rng.integers(width, size=edges)))
    types = rng.integers(3, size=edges)
    weights = rng.uniform(-1.5, 1.5, edges).astype(np.float32)
    numpy_dtype = dtype.as_numpy_dtype
    weights = weights.astype(numpy_dtype).astype(np.float32)
    spikes = rng.integers(0, 3, (batch, width)).astype(numpy_dtype)
    basis = rng.uniform(-0.8, 1.2, (3, 4)).astype(numpy_dtype)
    upstream = rng.uniform(-0.7, 0.7, (batch, posts, 4)).astype(numpy_dtype)
    connectivity = ops.build_csr_connectivity(indices, types, width, posts, 3)
    graph_def = graph_pb2.GraphDef.FromString(develop_graph_def(
        dtype, index_dtype, posts, edges, connectivity["n_pairs"],
    ))
    try:
        with tf.device("/GPU:0"):
            master = tf.constant(weights)
            csr = ops.reorder_csr_values(tf.cast(master, dtype), connectivity)

            @tf.function
            def run(z, w):
                feeds = {
                    "spikes": z, "master_weights": w, "metadata": connectivity["metadata_handle"],
                    "weights": csr, "basis": tf.constant(basis),
                    "spike_gradient_scale": tf.constant(1, dtype),
                    "active_rows": tf.zeros([0], tf.int64),
                    "incoming_pre_ids": tf.zeros([0], index_dtype),
                    "incoming_edge_ids": tf.zeros([0], index_dtype),
                    "incoming_types": tf.zeros([0], index_dtype),
                    "initial": tf.zeros([0], dtype),
                }
                current, = tf.graph_util.import_graph_def(
                    graph_def,
                    # Captured eager values must enter this FuncGraph before input_map.
                    input_map={f"{k}:0": tf.identity(v) for k, v in feeds.items()},
                    return_elements=["forward:0"], name="develop",
                )
                loss = tf.reduce_sum(
                    tf.cast(current, tf.float32) * tf.constant(upstream.reshape(-1, 4), tf.float32)
                )
                return (current, *tf.gradients(loss, [z, w]))

            current, dz, dw = run(tf.constant(spikes), master)
            graph = run.get_concrete_function(tf.constant(spikes), master).graph
            op_types = {op.type for op in graph.get_operations()}
        assert "DpointnetCsrSpikeForward" in op_types and "DpointnetCsrSpikeGrad" in op_types
        assert ops.CSR_MASTER_FORWARD_OP not in op_types
        expected = oracle(indices, types, weights, spikes, basis, upstream.astype(np.float32))
        tolerance = 2e-2 if dtype == tf.float16 else 2e-5
        for actual, reference in zip((current, dz, dw), expected):
            np.testing.assert_allclose(
                np.asarray(actual, np.float64), reference, rtol=tolerance, atol=tolerance,
            )
    finally:
        connectivity.close()


# Attributes a develop-built library's DpointnetCsrSpikeForward node carries; reading
# any other attribute raises like TensorFlow's Operation.get_attr.
def develop_op(compute_spike_gradient, compute_weight_gradient, op_type="DpointnetCsrSpikeForward",
               extra=None):
    values = {
        **DEVELOP_BOOL_DEFAULTS, "compute_spike_gradient": compute_spike_gradient,
        "compute_weight_gradient": compute_weight_gradient,
        "n_post": 5, "n_edges": 3, "n_pairs": 0, "Tindex": tf.uint32, **(extra or {}),
    }

    def get_attr(name):
        if name not in values:
            raise ValueError(f"Operation has no attr named '{name}'.")
        return values[name]

    inputs = [tf.zeros((2, 4), tf.float16), tf.zeros((3,), tf.float32)] + [None] * 8 + [
        tf.zeros((0,), tf.float16)
    ]
    return SimpleNamespace(type=op_type, get_attr=get_attr, inputs=inputs)


def develop_library(calls):
    """Gradient operators with develop's keyword signatures (no CSR-master attributes)."""

    def spike_grad(spikes, current_grad, metadata, weights, basis, scale, *, Tindex, n_post,
                   n_edges, n_pairs, use_packed_sm120_backward, write_csr_weight_gradient,
                   use_small_batch_backward, use_javier_batch32_backward):
        calls.append(("spike_grad", write_csr_weight_gradient))
        return tf.zeros((2, 4), tf.float16), tf.zeros((3,), tf.float32)

    def weight_grad(spikes, current_grad, metadata, basis, *, Tindex, n_post, n_edges, n_pairs,
                    use_packed_sm120_backward):
        calls.append(("weight_grad", None))
        return tf.zeros((3,), tf.float32)

    return SimpleNamespace(
        dpointnet_csr_spike_forward=lambda *args, **kwargs: None,
        dpointnet_csr_spike_grad=spike_grad,
        dpointnet_csr_weight_grad=weight_grad,
    )


@pytest.mark.parametrize("compute_spike_gradient,compute_weight_gradient,expected", [
    (True, True, [("spike_grad", False)]),
    (True, False, [("spike_grad", False)]),
    (False, True, [("weight_grad", None)]),
])
def test_gradient_never_reads_csr_master_attributes_of_develop_ops(
    monkeypatch, compute_spike_gradient, compute_weight_gradient, expected,
):
    calls = []
    monkeypatch.setattr(ops, "_OPS", develop_library(calls))
    grads = ops._fused_spike_currents_gradient(
        develop_op(compute_spike_gradient, compute_weight_gradient),
        tf.ones((10, 4), tf.float16),
    )
    assert calls == expected
    assert len(grads) == 11
    assert (grads[0] is not None) is compute_spike_gradient
    assert (grads[1] is not None) is compute_weight_gradient


def test_csr_master_proxy_without_uint8_attribute_defaults_to_wide_types(monkeypatch):
    seen = []

    def spike_grad_csr_master(*args, **kwargs):
        seen.append(kwargs)
        return tf.zeros((2, 4), tf.float16), tf.zeros((3,), tf.float32)

    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(
        dpointnet_csr_spike_grad_csr_master=spike_grad_csr_master,
    ))
    ops._fused_spike_currents_gradient(
        develop_op(True, True, op_type=ops.CSR_MASTER_FORWARD_OP), tf.ones((10, 4), tf.float16),
    )
    assert "uint8_types" not in seen[0] and seen[0]["write_csr_weight_gradient"] is True
    seen.clear()
    ops._fused_spike_currents_gradient(
        develop_op(True, True, op_type=ops.CSR_MASTER_FORWARD_OP, extra={"uint8_types": True}),
        tf.ones((10, 4), tf.float16),
    )
    assert seen[0]["uint8_types"] is True


@pytest.mark.parametrize("entry_points,available", [
    ((), False),
    (("dpointnet_master_projection",), False),
    (("dpointnet_master_projection_grad",), False),
    (("dpointnet_master_projection", "dpointnet_master_projection_grad"), True),
])
def test_reference_availability_requires_both_entry_points(monkeypatch, entry_points, available):
    from bmtk.simulator.dpointnet.custom_ops import master_projection
    # A develop-built _csr_spike_ops.so loads but has neither reference op.
    fake = SimpleNamespace(**{name: object() for name in entry_points})
    monkeypatch.setattr(master_projection, "_LIBRARY", fake)
    assert master_projection.reference_available() is available


def test_explicit_cpu_master_cell_rejects_a_library_without_reference_ops(monkeypatch):
    if tf.config.get_visible_devices("GPU"):
        pytest.skip("The CPU CSR-master reference is selected only with no visible GPU.")
    from bmtk.simulator.dpointnet.custom_ops import master_projection
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from test_precision_credit import make_cell
    monkeypatch.setattr(master_projection, "_LIBRARY", SimpleNamespace(
        dpointnet_csr_spike_forward=object(),
    ))
    network, inputs, arguments = make_cell(
        "nest", two_inputs=True, return_spec=True, train_recurrent=True,
        use_csr_master_weights=True,
    )
    for props in inputs.values():
        props["input_type"] = "spikes"
    with pytest.raises(RuntimeError, match="dpointnet_master_projection"):
        GLIF3Cell(network, inputs, **arguments)
