"""Matched optimized traversals: CPU contracts plus parent-only GPU oracles."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops
from bmtk.simulator.dpointnet.custom_ops import master_projection as primitive


KERNELS = (
    "DpointnetCombinedSweepKernel", "DpointnetCombinedAccumulateEdge",
    "CsrSpikeGradKernel", "CsrSpikeGradSmallBatchKernel",
    "DpointnetSpikeGradRowPerWarpHalfBatch32Kernel",
    "CsrSpikeGradPairBatch32Kernel", "CsrSpikeGradPairVariableBatchKernel",
    "CsrSpikeGradPairPackedBatch32Kernel",
)


@pytest.fixture(autouse=True)
def require_native_library():
    from bmtk.simulator.dpointnet.custom_ops.master_projection import reference_available
    if not reference_available():
        pytest.skip("Built native operators (and their CPU reference kernels) cannot be loaded here.")


def kernel_body(text, name):
    at = text.index(name + "(")
    start = text.index("{", at)
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


@pytest.mark.parametrize("name", KERNELS)
def test_optimized_kernel_bodies_are_unchanged(name):
    baseline = Path("/local2/bmtk/bmtk/simulator/dpointnet/custom_ops/csr_spike_ops.cu.cc")
    candidate = Path(ops.__file__).with_name("csr_spike_ops.cu.cc")
    assert kernel_body(candidate.read_text(), name) == kernel_body(baseline.read_text(), name)


def test_native_bindings_preserve_optimized_attributes():
    native = primitive.library()
    combined = inspect.signature(native.dpointnet_combined_sweep_csr_master).parameters
    for name in ("cooperative_rows", "fanout_rows", "gather_sources", "target_tile_size"):
        assert name in combined
    forward = inspect.signature(native.dpointnet_csr_spike_forward).parameters
    assert forward["csr_master_weights"].default is False
    weight_grad = inspect.signature(native.dpointnet_csr_weight_grad).parameters
    assert weight_grad["write_csr_weight_gradient"].default is False
    for name in ("dpointnet_csr_spike_grad_csr_master", "dpointnet_csr_spike_grad_accumulate_csr_master"):
        parameters = inspect.signature(getattr(native, name)).parameters
        for key in (
            "write_csr_weight_gradient", "use_small_batch_backward",
            "use_packed_sm120_backward", "use_javier_batch32_backward",
        ):
            assert key in parameters


@pytest.mark.parametrize("name", ["CsrWeightGradKernel", "CsrWeightGradPairPackedBatch32Kernel"])
def test_weight_only_kernel_changes_only_output_destination(name):
    import re
    baseline = Path("/local2/bmtk/bmtk/simulator/dpointnet/custom_ops/csr_spike_ops.cu.cc")
    candidate = Path(ops.__file__).with_name("csr_spike_ops.cu.cc")
    before = kernel_body(baseline.read_text(), name)
    after = kernel_body(candidate.read_text(), name)
    expression = "static_cast<int64_t>(edge_ids[edge])"
    replacement = (
        "(kWriteCsrGradient ? edge : static_cast<int64_t>(edge_ids[edge]))"
        if name == "CsrWeightGradKernel" else
        "kWriteCsrGradient ? static_cast<int64_t>(edge) : static_cast<int64_t>(edge_ids[edge])"
    )
    assert before.count(expression) == 1
    expected = before.replace(expression, replacement)
    assert re.sub(r"\s+", "", after) == re.sub(r"\s+", "", expected)


@pytest.mark.parametrize("carrier_enabled", [False, True])
def test_actual_native_while_vjp_graph_traces_without_gpu_execution(monkeypatch, carrier_enabled):
    native = primitive.library()
    monkeypatch.setattr(ops, "_OPS", native)
    monkeypatch.setattr(ops, "combined_sweep_available", lambda: True)
    metadata = tf.Variable(
        [0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 3, 4, 0, 3, 1, 2,
         0, 1, 2, 3, 0, 1, 2, 0, 0, 0, 0, 0],
        dtype=tf.int64, trainable=False,
    )
    conn = dict(
        metadata_handle=metadata.handle, index_dtype="int64", n_edges=4,
        n_pairs=4, n_post=3, n_sources=3, n_synapse_types=1,
        sorted_by_target=True, incoming_pre_ids=tf.zeros((0,), tf.int64),
        incoming_edge_ids=tf.zeros((0,), tf.int64),
        incoming_types=tf.zeros((0,), tf.int64),
    )
    # The existing accumulator ABI is uint32; both are traced without running native kernels.
    if carrier_enabled:
        metadata32 = tf.Variable(tf.cast(metadata, tf.uint32), trainable=False)
        conn["metadata_handle"] = metadata32.handle
        conn["index_dtype"] = "uint32"
        for key in ("incoming_pre_ids", "incoming_edge_ids", "incoming_types"):
            conn[key] = tf.zeros((0,), tf.uint32)
    recurrent = tf.Variable([0.11, -0.21, 0.31, 0.41])
    named = tf.Variable([0.51, -0.61, 0.71, 0.81])
    basis = tf.constant([[1., 0.5, 0.25, 0.125]], tf.float16)

    @tf.function
    def evaluate(z):
        with tf.GradientTape() as tape:
            tape.watch(z)
            initial_carrier = tf.identity(recurrent)

            def body(index, events, weight_carrier):
                if carrier_enabled:
                    rec_proxy, next_carrier = ops.fused_recurrent_weight_carry(
                        events, weight_carrier, recurrent, conn, basis, 3, 1.,
                        use_javier_batch32_backward=True, csr_master_weights=True,
                    )
                else:
                    rec_proxy = ops.fused_spike_currents(
                        events, recurrent, recurrent, conn, basis, 3, True,
                        vjp_only=True, use_packed_sm120_backward=False,
                        csr_master_weights=True,
                    )
                    next_carrier = weight_carrier
                input_proxy = ops.fused_spike_currents(
                    events, named, named, conn, basis, 3, False,
                    vjp_only=True, use_packed_sm120_backward=False,
                    csr_master_weights=True,
                )
                current = ops.fused_combined_currents(
                    [events, events], [recurrent, named], [conn, conn], basis,
                    [rec_proxy, input_proxy], csr_master_weights=True,
                )
                next_events = events + tf.reshape(tf.reduce_sum(current, axis=1), (2, 3)) * 0.001
                return index + 1, next_events, next_carrier

            _, result, _ = tf.while_loop(
                lambda index, *_: index < 3, body, (0, z, initial_carrier),
            )
            loss = tf.reduce_sum(tf.cast(result, tf.float32))
        return tape.gradient(loss, (z, recurrent, named))

    # Deliberately do not call the concrete function: no GPU compute is authorized here.
    graph = evaluate.get_concrete_function(tf.TensorSpec((2, 3), tf.float16)).graph.as_graph_def()
    nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
    names = {node.op for node in nodes}
    assert "DpointnetCombinedSweepCsrMaster" in names
    assert "DpointnetCsrWeightGrad" in names
    assert (
        "DpointnetCsrSpikeGradAccumulateCsrMaster" if carrier_enabled
        else "DpointnetCsrSpikeGradCsrMaster"
    ) in names
    forward = [node for node in nodes if node.op == "DpointnetCsrSpikeForward"]
    assert forward and all(node.attr["csr_master_weights"].b for node in forward)
    assert all(tf.as_dtype(node.attr["W"].type) == tf.float32 for node in forward)
    assert not names.intersection({"DpointnetCsrReorder", "DpointnetCsrRestore"})
    weight_vjps = [node for node in nodes if node.op == "DpointnetCsrWeightGrad"]
    assert all(node.attr["write_csr_weight_gradient"].b for node in weight_vjps)


@pytest.mark.parametrize("compiled", [False, True])
def test_carrier_boundary_keeps_live_canonical_weights_with_cpu_native_oracle(monkeypatch, compiled):
    from test_master_projection import fixture, oracle
    indices, types, w, z, basis, dy, metadata = fixture(np.float16, tf.int64)
    resource = tf.Variable(metadata, trainable=False)
    conn = {
        "metadata_handle": resource.handle, "index_dtype": "int64",
        "n_post": 5, "n_edges": len(w), "n_pairs": 1,
        "n_sources": z.shape[1], "n_synapse_types": len(basis),
        "incoming_pre_ids": tf.zeros((0,), tf.int64),
        "incoming_edge_ids": tf.zeros((0,), tf.int64),
        "incoming_types": tf.zeros((0,), tf.int64),
    }
    native = primitive.library()
    order = metadata.numpy()[-len(w):]
    seen = []

    def forward(spikes, master, routing, weights, coefficients, scale, *unused, **options):
        assert options["vjp_only"] and options["csr_master_weights"]
        assert weights.dtype == tf.float32
        # Match the producer's resource input position for the retained carrier wrapper.
        return tf.raw_ops.IdentityN(input=[
            tf.zeros((z.shape[0] * 5, 4), spikes.dtype), master, routing,
        ])[0]

    def accumulate(spikes, current_grad, routing, weights, coefficients, scale, seed, **options):
        seen.append(options["use_javier_batch32_backward"])
        values = tf.raw_ops.ReadVariableOp(resource=routing, dtype=tf.int64)
        dz, dw = native.dpointnet_master_projection_grad(
            spikes, weights, values, coefficients, tf.cast(current_grad, tf.float32),
            n_post=5, n_edges=len(w),
        )
        return tf.cast(dz * tf.cast(scale, tf.float32), spikes.dtype), seed + tf.gather(dw, order)

    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(
        dpointnet_csr_spike_forward=forward,
        dpointnet_csr_spike_grad_accumulate=lambda *a, **k: None,
        dpointnet_csr_spike_grad_accumulate_csr_master=accumulate,
    ))
    master = tf.Variable(w)
    spikes = tf.Variable(z)

    def evaluate():
        with tf.GradientTape() as tape:
            carrier = tf.identity(master)
            proxy, final_carrier = ops.fused_recurrent_weight_carry(
                spikes, carrier, master, conn, tf.constant(basis), 5, 1.0,
                csr_master_weights=True, use_javier_batch32_backward=True,
            )
            objective = tf.reduce_sum(tf.cast(proxy, tf.float32) * dy.astype(np.float16).reshape(-1, 4))
            objective += tf.reduce_sum(final_carrier * 0.125)
        return tape.gradient(objective, [spikes, master])

    run = tf.function(evaluate) if compiled else evaluate
    for offset in (0.0, 0.27):
        master.assign(w + offset)
        dz, dw = run()
        expected = oracle(indices, types, master.numpy(), z, basis, dy.astype(np.float16))
        np.testing.assert_allclose(dz, expected[1], rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(dw, expected[2][order] + 0.125, rtol=6e-6, atol=6e-6)
    assert seen and all(seen)


@pytest.mark.parametrize("dtype,batch,tile,mode,packed,small", [
    (np.float16, 1, 256, "cooperative", False, False),
    (np.float16, 3, 2048, "fanout", False, True),
    (np.float16, 32, 2048, "all", False, False),
    (np.float16, 32, 5120, "all", True, False),
    (np.float32, 3, 2048, "fanout", False, True),
    (np.float32, 32, 2048, "all", False, False),
])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
def test_gpu_matched_optimized_forward_and_master_vjps(dtype, batch, tile, mode, packed, small, index_dtype, monkeypatch):
    if not ops.combined_sweep_available():
        pytest.skip("Parent-only GPU execution; rebuilt optimized operators required")
    if packed and index_dtype != "uint32":
        pytest.skip("Existing packed backward requires uint32 metadata")
    from test_combined_sweep import fixture
    from test_master_projection import oracle
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: tf.as_dtype(index_dtype))
    basis, sources = fixture(dtype, batch, high_fanout=mode in ("fanout", "all"))
    if mode == "cooperative":
        indices, types, weights, spikes, conn = sources[0]
        conn.close()
        unique, first = np.unique(indices, axis=0, return_index=True)
        conn = ops.build_csr_connectivity(
            unique, types[first], spikes.shape[1], 521, len(basis),
            build_compact_pairs=True, sort_by_target=True,
        )
        assert conn["has_repeated_targets"] is False
        sources[0] = unique, types[first], weights[first], spikes, conn
    rng = np.random.default_rng(13)
    upstream = rng.uniform(-0.5, 0.5, (batch, 521, 4)).astype(dtype)
    zs = [tf.Variable(source[3]) for source in sources]
    from bmtk.simulator.dpointnet.csr_master import CsrMasterLayout
    layouts = [CsrMasterLayout(source[4]) for source in sources]
    masters = [
        tf.Variable(layout.to_live(source[2].astype(np.float32) + np.float32(0.00031)))
        for source, layout in zip(sources, layouts)
    ]
    shadows = [tf.cast(w, zs[0].dtype) for w in masters]
    schedule = dict(
        cooperative_rows=mode in ("cooperative", "all"),
        fanout_rows=mode in ("fanout", "all"),
        gather_sources=mode == "all", target_tile_size=tile,
    )

    def evaluate(canonical):
        with tf.GradientTape() as tape:
            proxies = [
                ops.fused_spike_currents(
                    z, w, w if canonical else shadow, src[4], tf.constant(basis), 521,
                    compute_spike_gradient=True, vjp_only=True,
                    use_packed_sm120_backward=packed,
                    use_small_batch_backward=small,
                    **({"csr_master_weights": True} if canonical else {}),
                )
                for z, w, shadow, src in zip(zs, masters, shadows, sources)
            ]
            out = ops.fused_combined_currents(
                zs, masters if canonical else shadows, [src[4] for src in sources],
                tf.constant(basis), proxies, csr_master_weights=canonical, **schedule,
            )
            loss = tf.reduce_sum(tf.cast(out, tf.float32) * upstream.reshape(-1, 4))
        return out, tape.gradient(loss, zs + masters)

    candidate = tf.function(lambda: evaluate(True))
    reference = tf.function(lambda: evaluate(False))
    try:
        for offset in (0.0, 0.19):
            if offset:
                for master, shadow, src in zip(masters, shadows, sources):
                    master.assign_add(tf.ones_like(master) * offset)
                    # The legacy control must see the same assigned master at its own precision.
                shadows[:] = [tf.cast(w, zs[0].dtype) for w in masters]
                reference = tf.function(lambda: evaluate(False))
            actual, gradients = candidate()
            control, control_gradients = reference()
            expected = [
                oracle(src[0], src[1], layout.to_canonical(master).numpy(), z.numpy(), basis, upstream)
                for src, master, z, layout in zip(sources, masters, zs, layouts)
            ]
            tolerance = 2e-3 if dtype == np.float16 else 1e-5
            np.testing.assert_allclose(actual, sum(item[0] for item in expected),
                                       rtol=tolerance, atol=tolerance)
            for got, want in zip(gradients, [item[1] for item in expected] + [
                layout.to_live(item[2]) for item, layout in zip(expected, layouts)
            ]):
                np.testing.assert_allclose(got, want, rtol=0.02 if dtype == np.float16 else 2e-5,
                                           atol=0.015 if dtype == np.float16 else 2e-5)
            # Identical optimized arithmetic, allowing true master-versus-half operand differences.
            np.testing.assert_allclose(actual, control, rtol=0.01, atol=0.02)
            assert all(g is not None for g in control_gradients)
    finally:
        for src in sources:
            src[4].close()


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("javier", [False, True])
def test_gpu_optimized_accumulator_reads_canonical_masters(batch, javier):
    if not ops.combined_sweep_available():
        pytest.skip("Parent-only GPU execution; rebuilt optimized operators required")
    from test_combined_sweep import fixture
    from test_master_projection import oracle
    basis, sources = fixture(np.float16, batch)
    indices, types, weights, spikes, conn = sources[0]
    canonical = tf.constant(weights.astype(np.float32) + np.float32(0.00031))
    master = ops.reorder_csr_values(canonical, conn)
    dy = np.random.default_rng(81).uniform(-0.5, 0.5, (batch, 521, 4)).astype(np.float16)
    try:
        dz, csr_dw = ops._OPS.dpointnet_csr_spike_grad_accumulate_csr_master(
            tf.constant(spikes), tf.constant(dy.reshape(-1, 4)),
            conn["metadata_handle"], master, tf.constant(basis), tf.constant(1., tf.float16),
            tf.fill((len(weights),), 0.125),
            Tindex=tf.uint32, n_post=521, n_edges=len(weights), n_pairs=conn["n_pairs"],
            use_javier_batch32_backward=javier,
        )
        dw = ops.restore_csr_values(csr_dw, conn)
        expected = oracle(indices, types, canonical.numpy(), spikes, basis, dy)
        np.testing.assert_allclose(dz, expected[1], rtol=0.02, atol=0.015)
        np.testing.assert_allclose(dw, expected[2] + 0.125, rtol=0.02, atol=0.015)
        assert np.any(dz.numpy()[:, 0] != 0)
    finally:
        for src in sources:
            src[4].close()


@pytest.mark.parametrize("dtype,batch,index_dtype,packed", [
    (np.float16, 1, "uint32", False),
    (np.float16, 3, "int64", False),
    (np.float16, 32, "uint32", False),
    (np.float16, 32, "uint32", True),
    (np.float32, 1, "int64", False),
    (np.float32, 32, "uint32", False),
])
def test_gpu_weight_only_vjp_writes_live_csr_directly(dtype, batch, index_dtype, packed, monkeypatch):
    if not ops.combined_sweep_available():
        pytest.skip("GPU and rebuilt native direct-CSR weight gradients required")
    from test_combined_sweep import fixture
    from test_master_projection import oracle
    from bmtk.simulator.dpointnet.csr_master import CsrMasterLayout
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: tf.as_dtype(index_dtype))
    basis, sources = fixture(dtype, batch)
    upstream = np.random.default_rng(79).uniform(-0.5, 0.5, (batch, 521, 4)).astype(dtype)
    try:
        for indices, types, weights, spikes, conn in sources:
            operands = (
                tf.constant(spikes), tf.constant(upstream.reshape(-1, 4)),
                conn["metadata_handle"], tf.constant(basis),
            )
            arguments = dict(
                Tindex=tf.as_dtype(index_dtype), n_post=521,
                n_edges=conn["n_edges"], n_pairs=conn["n_pairs"],
                use_packed_sm120_backward=packed,
            )
            canonical = ops._OPS.dpointnet_csr_weight_grad(*operands, **arguments)
            actual = ops._OPS.dpointnet_csr_weight_grad(
                *operands, **arguments, write_csr_weight_gradient=True,
            )
            expected = oracle(indices, types, weights, spikes, basis, upstream)[2]
            tolerance = 2e-5
            np.testing.assert_allclose(canonical, expected, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(
                actual, CsrMasterLayout(conn).to_live(expected), rtol=tolerance, atol=tolerance,
            )
    finally:
        for source in sources:
            source[4].close()


@pytest.mark.parametrize("carrier,frozen_named", [(False, False), (True, False), (True, True)])
def test_gpu_auto_cell_replay_update_and_explicit_restore(monkeypatch, tmp_path, carrier, frozen_named):
    if not ops.combined_sweep_available():
        pytest.skip("GPU and rebuilt optimized operators required")
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet import acceleration
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.segmented_recompute import (
        FullBPTTGradientRunner, SegmentedRecomputeRunner,
    )

    def construct(enabled):
        network, inputs, arguments = make_cell(
            "nest", fused=True, two_inputs=True, return_spec=True,
            train_recurrent=True, batch_size=32, acceleration_profile="auto",
            use_csr_master_weights=enabled, use_uint8_synapse_types=False,
            use_adaptive_routing=False,
            _canonical_gradient_boundary=True, use_fused_recurrent_accumulation=carrier,
        )
        for props in inputs.values():
            props["input_type"] = "spikes"
        inputs["background"]["options"]["trainable"] = not frozen_named
        arguments["_performance_auto_context"] = acceleration.performance_projection_context(
            network, inputs, dt=1.0, max_delay=5.0, standard_cell=True, single_replica=True,
        )
        return GLIF3Cell(network, inputs, **arguments)

    control = construct(False)
    cell = construct(True)
    differing = {"use_csr_master_weights"}
    assert {k: v for k, v in cell.acceleration_report["selected"].items() if k not in differing} == {
        k: v for k, v in control.acceleration_report["selected"].items() if k not in differing
    }
    assert cell.acceleration_report["csr_master_projection"] == "optimized_combined_csr_fp32_master"
    assert control.acceleration_report["csr_master_weights"] == "disabled"
    assert cell._use_fused_cuda and cell._use_combined_current_sweep
    assert cell._use_direct_csr_recurrent_gradient
    assert cell.acceleration_report["csr_master_weights"] == "direct_csr_fp32_master_host_lifecycle_v1"
    assert "CPU:" in cell.recurrent_indices.device
    assert isinstance(cell.recurrent_master_layout.csr_to_canonical, np.ndarray)
    assert cell.recurrent_master_layout.csr_to_canonical.dtype == np.uint32
    assert not cell.recurrent_fused_connectivity["device_edge_ids"]
    assert cell.use_fused_recurrent_accumulation == carrier
    masters = [cell.recurrent_weight_values] + [
        net["input_weight_values"] for net in cell.inputs.values()
    ]
    control_masters = [control.recurrent_weight_values] + [
        net["input_weight_values"] for net in control.inputs.values()
    ]
    layouts = [cell.recurrent_master_layout] + [
        net["master_layout"] for net in cell.inputs.values()
    ]
    for master, reference, layout in zip(masters, control_masters, layouts):
        np.testing.assert_array_equal(master, layout.to_live(reference))
        np.testing.assert_array_equal(
            master.constraint.condition, layout.to_live(reference.constraint.condition)
        )
    assert cell.recurrent_weight_values_compute is masters[0]
    assert cell.recurrent_csr_weight_values_compute is None
    assert control.recurrent_csr_weight_values_compute is not None
    for net in cell.inputs.values():
        assert net["input_weight_values_compute"] is net["input_weight_values"]
        assert net["csr_weight_values_compute"] is None
    assert all(master.dtype == "float32" for master in masters)
    trainable = [master for master in masters if master.trainable]
    state = list(cell.zero_state(32, cell.compute_dtype))
    state[1] = tf.tile(tf.constant([[1.1, 0.8]], state[1].dtype), [32, 1])
    width = int(cell.inputs_idx[-1])
    sequence = tf.ones((32, 7, width), cell.compute_dtype)
    sequence_input = tf.keras.Input(shape=(None, width), dtype=cell.compute_dtype)
    state_inputs = [tf.keras.Input(shape=x.shape[1:], dtype=x.dtype) for x in state]
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    layer._autocast = False
    out = layer(sequence_input, initial_state=state_inputs)
    core = tf.keras.Model([sequence_input, *state_inputs], [*out[0], *out[1:]])
    from bmtk.simulator.dpointnet.training import TrainingEngine, TrainingParameters
    from bmtk.simulator.dpointnet.loss_functions import weight_regularization as wr
    import bmtk.simulator.dpointnet.cell_models.glif3_cell as cell_module
    monkeypatch.setattr(
        cell_module, "restore_csr_values",
        lambda *args, **kwargs: pytest.fail("CSR-live gradients must not be canonical-restored"),
    )
    engines = []
    for checkpointing in (False, True):
        engine = TrainingEngine(
            SimpleNamespace(cell=cell, extractor_model=core), 1, 1,
            training_approach="single", gradient_checkpointing=checkpointing,
            gradient_checkpoint_chunk_size=3, pack_spike_checkpoints=True,
        )
        engine._parameters = [TrainingParameters("fixture", batch_size=32, seq_len=7)]
        engine.prepare_gradient_checkpointing()
        engines.append(engine)
    full, segment = [engine._extractor_forward for engine in engines]
    assert segment.variable_gradient_transform is None
    monkeypatch.setattr(
        wr, "_connection_type_ids",
        lambda *args, **kwargs: np.arange(int(masters[0].shape[0]), dtype=np.int64),
    )
    regularizer = wr.EMDWeightRegularization(
        SimpleNamespace(cell=cell, recurrent_network={}), cost=0.3,
    )
    assert not np.array_equal(
        cell.recurrent_master_layout.csr_to_canonical,
        np.arange(int(masters[0].shape[0])),
    )
    for master in trainable:
        master.assign(master * 1.02)

    def evaluate(runner):
        with tf.GradientTape() as tape:
            values = core([sequence, *state]) if runner is None else runner(sequence, state)
            loss = (
                tf.reduce_mean(tf.cast(values[0], tf.float32))
                + tf.reduce_mean(values[1]) + regularizer()
            )
        return values, tape.gradient(loss, trainable)

    full_run = tf.function(lambda: evaluate(full))
    segment_run = tf.function(lambda: evaluate(segment))
    raw_run = tf.function(lambda: evaluate(None))
    graph = segment_run.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
    assert not any(node.op in ("DpointnetCsrReorder", "DpointnetCsrRestore") for node in nodes)
    weight_vjps = [node for node in nodes if node.op == "DpointnetCsrWeightGrad"]
    if not frozen_named:
        assert weight_vjps
    assert all(node.attr["write_csr_weight_gradient"].b for node in weight_vjps)

    def check():
        actual = segment_run()
        for expected in (full_run(), raw_run()):
            for got, want in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
                assert got is not None and want is not None
                np.testing.assert_allclose(got, want, rtol=3e-3, atol=2e-3)
        for gradient in actual[1]:
            assert gradient.dtype == tf.float32
            assert np.all(np.isfinite(gradient))
            assert np.any(gradient.numpy() != 0)
        return actual

    optimizer = tf.keras.optimizers.Adam(0.001)
    optimizer.build(trainable)
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, noise_seed=cell.noise_seed, noise_stream=cell.noise_stream,
        **{f"master_{i}": master.value for i, master in enumerate(masters)},
    )
    @tf.function
    def update():
        values, gradients = evaluate(segment)
        optimizer.apply_gradients(zip(gradients, trainable))
        return values

    check()
    update()
    cell.refresh_recurrent_weight_shadow()
    first = check()
    path = checkpoint.save(str(tmp_path / "native_all_masters"))
    update()
    for master in trainable:
        assert np.all(np.where(
            master.constraint.condition.numpy(), master.numpy() >= 0, master.numpy() <= 0,
        ))
    cell.refresh_recurrent_weight_shadow()
    expected_next = check()
    optimizer_values = [v.numpy().copy() for v in optimizer.variables]
    for master in masters:
        master.assign(master * 1.2)
    cell.noise_seed.assign_add(1)
    cell.noise_stream.assign_add(1)
    assigned = check()
    assert not np.array_equal(first[0][1], assigned[0][1])
    checkpoint.restore(path).assert_consumed()
    restored = check()
    for got, want in zip(tf.nest.flatten(restored), tf.nest.flatten(first)):
        np.testing.assert_array_equal(got, want)
    assert int(optimizer.iterations.numpy()) == 1
    assert int(cell.noise_stream.numpy()) == 0
    update()
    actual_next = check()
    for got, want in zip(tf.nest.flatten(actual_next), tf.nest.flatten(expected_next)):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(optimizer.variables, optimizer_values):
        np.testing.assert_array_equal(got, want)
