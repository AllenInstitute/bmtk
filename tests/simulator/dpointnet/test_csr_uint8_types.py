"""Safe mixed-width native metadata, without a widened per-edge type shadow."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops
from bmtk.simulator.dpointnet.csr_master import CsrMasterLayout


NAMES = (
    "dpointnet_csr_spike_forward", "dpointnet_csr_weight_grad",
    "dpointnet_csr_spike_grad_csr_master",
    "dpointnet_csr_spike_grad_accumulate_csr_master",
    "dpointnet_combined_sweep_csr_master",
)


def capability(*, uint8_types=False):
    return uint8_types


@pytest.fixture
def host_capability(monkeypatch):
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(**{name: capability for name in NAMES}))


@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
@pytest.mark.parametrize("edges", [0, 1, 3, 4, 5, 8, 41])
@pytest.mark.parametrize("pairs", [False, True])
def test_actual_resource_sections_alignment_and_single_ownership(host_capability, monkeypatch, index, edges, pairs):
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: index)
    endpoints = np.column_stack((np.arange(edges) % 3, np.arange(edges) % 7))
    types = (np.arange(edges) % 2 * 255).astype(np.int64)
    arguments = dict(build_compact_pairs=pairs, sort_by_target=True, host_lifecycle=True)
    baseline = ops.build_csr_connectivity(endpoints, types, 7, 3, 256, **arguments)
    typed = ops.build_csr_connectivity(endpoints, types, 7, 3, 256, uint8_types=True, **arguments)
    try:
        assert len(typed.metadata_resource_handles) == 1
        assert typed.metadata_resource_handles[0] is typed["metadata_handle"]
        for conn in (baseline, typed):
            specs = conn["metadata_resource_specs"]
            assert specs is conn.metadata_resource_specs
            assert tuple(spec["handle"] for spec in specs) == conn.metadata_resource_handles
            assert len({spec["handle"].ref() for spec in specs}) == len(specs)
            for spec in specs:
                actual_spec = tf.raw_ops.ReadVariableOp(
                    resource=spec["handle"], dtype=tf.as_dtype(spec["dtype"]),
                )
                assert tuple(actual_spec.shape) == spec["shape"]
                assert actual_spec.dtype.size * int(np.prod(actual_spec.shape)) == spec["payload_bytes"]
                assert spec["payload_bytes"] == spec["logical_payload_bytes"] + spec["alignment_bytes"]
                assert spec["payload_bytes"] == conn["metadata_bytes"]
                assert spec["role"] == "csr_metadata" and spec["schema"] == conn["metadata_schema"]
                assert spec["type_dtype"] == conn["type_dtype"]
                assert spec["type_payload_bytes"] == edges * (1 if conn["uint8_types"] else index.size)
                assert spec["type_logical_min"] == (int(types.min()) if edges else None)
                assert spec["type_logical_max"] == (int(types.max()) if edges else None)
            with monkeypatch.context() as scoped:
                scoped.setattr(tf.raw_ops, "ReadVariableOp", lambda **kw: pytest.fail("Census read a tensor"))
                scoped.setattr(tf, "identity", lambda *a, **kw: pytest.fail("Census copied a tensor"))
                assert conn.metadata_resource_specs is specs
                assert conn["metadata_resource_specs"] is specs
        old = tf.raw_ops.ReadVariableOp(resource=baseline["metadata_handle"], dtype=index).numpy()
        actual = tf.raw_ops.ReadVariableOp(resource=typed["metadata_handle"], dtype=index).numpy()
        assert tuple(actual.shape) == typed["metadata_shape"]
        assert actual.nbytes == typed["metadata_bytes"]
        offset = typed["type_byte_offset"]
        assert offset % index.size == 0
        np.testing.assert_array_equal(actual[:edges], old[:edges])
        np.testing.assert_array_equal(actual[edges:offset // index.size], old[2*edges:])
        np.testing.assert_array_equal(actual.view(np.uint8)[offset:offset+edges], old[edges:2*edges])
        np.testing.assert_array_equal(actual.view(np.uint8)[offset+edges:], 0)
        assert baseline["metadata_bytes"] - typed["metadata_bytes"] == (
            (index.size - 1) * edges - (-edges) % index.size
        )
        assert typed["type_dtype"] == "uint8"
        assert typed["metadata_payload_bytes"] + typed["metadata_padding_bytes"] == actual.nbytes
        assert typed["metadata_schema"] == "csr_live_uint8_type_tail_v1"
        np.testing.assert_array_equal(typed["host_edge_order"], baseline["host_edge_order"])
        for consumer in (ops.reorder_csr_values, ops.restore_csr_values):
            with pytest.raises(ValueError, match="canonical edge IDs"):
                consumer(tf.zeros(edges), typed)
    finally:
        typed.close()
        baseline.close()


@pytest.mark.parametrize("values,error", [
    ([-1], ValueError), ([256], ValueError), ([2**63], ValueError),
    ([1.5], TypeError), ([np.nan], TypeError), ([np.inf], TypeError),
    (["1"], TypeError), ([True], TypeError), ([1+0j], TypeError),
])
def test_reject_before_narrow_or_resource_allocation(host_capability, monkeypatch, values, error):
    monkeypatch.setattr(ops, "_create_metadata_resource", lambda *a: pytest.fail("allocated invalid metadata"))
    with pytest.raises(error, match="integer dtype|\\[0, 255\\]"):
        ops.build_csr_connectivity([[0, 0]], values, 1, 1, 512, host_lifecycle=True, uint8_types=True)


def test_larger_valid_types_remain_wide_when_disabled(host_capability):
    conn = ops.build_csr_connectivity([[0, 0]], [256], 1, 1, 512, host_lifecycle=True)
    try:
        raw = tf.raw_ops.ReadVariableOp(resource=conn["metadata_handle"], dtype=tf.as_dtype(conn["index_dtype"]))
        assert raw.numpy()[1] == 256
        assert conn["type_dtype"] == conn["index_dtype"]
        spec, = conn.metadata_resource_specs
        assert spec["type_logical_min"] == spec["type_logical_max"] == 256
        assert spec["type_payload_bytes"] == raw.dtype.size
    finally:
        conn.close()


@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
@pytest.mark.parametrize("host_lifecycle,typed", [(False, False), (True, False), (True, True)])
def test_specs_enumerate_each_owned_resource(host_capability, monkeypatch, index, host_lifecycle, typed):
    from bmtk.simulator.dpointnet import distribution

    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: index)
    monkeypatch.setattr(distribution, "data_parallel_worker_devices", lambda: ("/CPU:0", "/CPU:0"))
    conn = ops.build_csr_connectivity(
        [[0, 0]], [255], 1, 1, 256, host_lifecycle=host_lifecycle, uint8_types=typed,
    )
    try:
        specs = conn.metadata_resource_specs
        assert len(specs) == 2
        assert len({spec["handle"].ref() for spec in specs}) == 2
        assert tuple(spec["handle"] for spec in specs) == conn.metadata_resource_handles
        for spec in specs:
            raw = tf.raw_ops.ReadVariableOp(resource=spec["handle"], dtype=tf.as_dtype(spec["dtype"]))
            assert tuple(raw.shape) == spec["shape"]
            assert raw.numpy().nbytes == spec["payload_bytes"]
            assert spec["alignment_bytes"] == ((-1) % index.size if typed else 0)
            assert spec["type_payload_bytes"] == (1 if typed else index.size)
            assert spec["type_logical_min"] == spec["type_logical_max"] == 255
    finally:
        conn.close()


@pytest.mark.parametrize("value", [None, 0, 1, "0", "auto"])
def test_non_boolean_low_level_requests_reject(host_capability, value):
    with pytest.raises(TypeError, match="explicit boolean"):
        ops.build_csr_connectivity([[0, 0]], [0], 1, 1, 1, host_lifecycle=True, uint8_types=value)


def test_invalid_flag_and_cpu_composition_reject(monkeypatch):
    from test_precision_credit import make_cell
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_UINT8_TYPES", "auto")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        make_cell()
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_UINT8_TYPES", "1")
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_FP32_MASTERS", "1")
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_HOST_LIFECYCLE", "1")
    with pytest.raises(ValueError, match="require native fused CUDA"):
        make_cell("nest")


@pytest.mark.parametrize("missing", NAMES)
def test_old_partial_native_libraries_reject_explicit_typed_request(monkeypatch, missing):
    library = {name: capability for name in NAMES}
    library[missing] = lambda: None
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(**library))
    with pytest.raises(RuntimeError, match="Rebuild both CUDA operators"):
        ops.build_csr_connectivity([[0, 0]], [0], 1, 1, 1, host_lifecycle=True, uint8_types=True)
    assert ops._uint8_type_options({}) == {}


@pytest.mark.parametrize("csr,host", [(0, 0), (0, 1), (1, 0)])
def test_flag_composition_rejects_unsupported_modes(monkeypatch, csr, host):
    from test_precision_credit import make_cell
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_UINT8_TYPES", "1")
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_FP32_MASTERS", str(csr))
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_HOST_LIFECYCLE", str(host))
    with pytest.raises(ValueError, match="require CSR FP32 masters and host lifecycle"):
        make_cell()


def native():
    if not tf.config.get_visible_devices("GPU") or not ops.combined_sweep_available():
        pytest.skip("native GPU test requires the isolated rebuilt library")
    for name in NAMES:
        assert "uint8_types" in inspect.signature(getattr(ops._OPS, name)).parameters


@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [1, 3, 32])
@pytest.mark.parametrize("empty", [False, True])
def test_gpu_fp64_forward_spike_weight_and_queue_gather_oracles(monkeypatch, index, dtype, batch, empty):
    native()
    from test_master_projection import oracle
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *a: index)
    rng = np.random.default_rng(818)
    edges, width, posts = (0 if empty else 41), 7, 5
    endpoints = np.column_stack((rng.integers(posts, size=edges), rng.integers(width, size=edges)))
    types = rng.choice(np.array([0, 1, 255], np.int64), size=edges)
    weights = rng.uniform(-0.7, 0.9, edges).astype(np.float32)
    events = rng.integers(0, 4, (batch, width)).astype(dtype)
    events[:, 0] = 0
    basis = rng.uniform(-0.3, 0.8, (256, 4)).astype(dtype)
    dy = rng.uniform(-0.4, 0.5, (batch, posts, 4)).astype(dtype)
    args = dict(build_compact_pairs=True, sort_by_target=True, host_lifecycle=True)
    conns = [ops.build_csr_connectivity(endpoints, types, width, posts, 256, uint8_types=t, **args) for t in (False, True)]
    try:
        results = []
        for conn in conns:
            mapping = CsrMasterLayout(conn, host_lifecycle=True)
            w = tf.Variable(mapping.host_to_live(weights))
            z = tf.Variable(events)
            def evaluate():
                with tf.GradientTape() as tape:
                    proxy = ops.fused_spike_currents(
                        z, w, w, conn, tf.constant(basis), posts, True,
                        vjp_only=True, csr_master_weights=True, use_packed_sm120_backward=False,
                        write_csr_weight_gradient=True, use_device_active_queue_forward=True,
                    )
                    out = ops.fused_combined_currents(
                        [z, z], [w, w], [conn, conn], tf.constant(basis), [proxy, proxy],
                        csr_master_weights=True, gather_sources=True, fanout_rows=True, cooperative_rows=True,
                    )
                    loss = tf.reduce_sum(tf.cast(out, tf.float32) * dy.reshape(-1, 4))
                return out, *tape.gradient(loss, (z, w))
            run = tf.function(evaluate)
            got = run()
            graph = run.get_concrete_function().graph.as_graph_def()
            nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
            consumers = [n for n in nodes if n.op in (
                "DpointnetCsrSpikeForward", "DpointnetCsrSpikeGradCsrMaster", "DpointnetCombinedSweepCsrMaster",
            )]
            assert len(consumers) == 3
            assert all(n.attr["uint8_types"].b == conn["uint8_types"] for n in consumers)
            assert all(n.attr["target_tile_size"].i == 2048 for n in consumers if n.op == "DpointnetCombinedSweepCsrMaster")
            expected = oracle(endpoints, types, weights, events, basis, dy)
            tol = 0.02 if dtype == np.float16 else 2e-5
            for actual, ref in zip(got, (2*expected[0], 2*expected[1], mapping.host_to_live(2*expected[2]))):
                np.testing.assert_allclose(actual, ref, rtol=tol, atol=tol)
            if edges:
                assert np.any(got[1].numpy()[:, 0] != 0), "silent connected source lost credit"
            options = ops._uint8_type_options(conn)
            dw = ops._OPS.dpointnet_csr_weight_grad(
                tf.constant(events), tf.constant(dy.reshape(-1, 4)), conn["metadata_handle"],
                tf.constant(basis), Tindex=index, n_post=posts, n_edges=edges, n_pairs=conn["n_pairs"],
                write_csr_weight_gradient=True, **options,
            )
            np.testing.assert_allclose(dw, mapping.host_to_live(expected[2]), rtol=2e-5, atol=2e-5)
            results.append(got)
        for a, b in zip(results[0], results[1]):
            np.testing.assert_allclose(a, b, rtol=0.002 if dtype == np.float16 else 2e-6, atol=0.004 if dtype == np.float16 else 2e-6)
    finally:
        for conn in conns:
            conn.close()


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [1, 3, 32])
@pytest.mark.parametrize("spike_credit", [False, True])
def test_gpu_seeded_carrier_and_weight_only_credit(monkeypatch, dtype, batch, spike_credit):
    native()
    if dtype == np.float32 and not spike_credit:
        pytest.skip("existing weight-only carrier ABI is FP16 only")
    from test_master_projection import oracle
    rng = np.random.default_rng(119)
    indices = np.array([[0, 0], [1, 1], [1, 1], [0, 2], [1, 0]])
    types = np.array([0, 255, 255, 1, 0])
    weights = np.array([0.11, -0.23, 0.37, -0.41, 0.59], np.float32)
    z = rng.integers(0, 4, (batch, 3)).astype(dtype)
    z[:, 0] = 0
    basis = rng.uniform(-0.3, 0.8, (256, 4)).astype(dtype)
    dy = rng.uniform(-0.4, 0.5, (batch, 2, 4)).astype(dtype)
    conn = ops.build_csr_connectivity(indices, types, 3, 2, 256, build_compact_pairs=True, host_lifecycle=True, uint8_types=True)
    try:
        mapping = CsrMasterLayout(conn, host_lifecycle=True)
        seed = np.linspace(-0.3, 0.4, 5).astype(np.float32)
        dz, dw = ops._OPS.dpointnet_csr_spike_grad_accumulate_csr_master(
            tf.constant(z), tf.constant(dy.reshape(-1, 4)), conn["metadata_handle"],
            tf.constant(mapping.host_to_live(weights)), tf.constant(basis), tf.constant(0.25, tf.as_dtype(dtype)),
            tf.constant(seed), Tindex=tf.uint32, n_post=2, n_edges=5, n_pairs=conn["n_pairs"],
            uint8_types=True, compute_spike_gradient=spike_credit, use_javier_batch32_backward=True,
        )
        expected = oracle(indices, types, weights, z, basis, dy)
        tol = 0.002 if dtype == np.float16 else 2e-5
        np.testing.assert_allclose(dw, mapping.host_to_live(expected[2])+seed, rtol=tol, atol=tol)
        np.testing.assert_allclose(dz, expected[1]*0.25 if spike_credit else np.zeros_like(z), rtol=tol, atol=tol)
    finally:
        conn.close()


@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
@pytest.mark.parametrize("bad", ["short", "long", "rank", "dtype", "overflow"])
def test_gpu_native_rejects_malformed_resource_before_pointer_reads(monkeypatch, index, bad):
    native()
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *a: index)
    conn = ops.build_csr_connectivity([[0, 0]], [255], 1, 1, 256, host_lifecycle=True, uint8_types=True)
    try:
        raw = tf.raw_ops.ReadVariableOp(resource=conn["metadata_handle"], dtype=index)
        data = raw.numpy()
        if bad == "short":
            data = data[:-1]
        elif bad == "long":
            data = np.r_[data, data[:1]]
        elif bad == "rank":
            data = data.reshape(1, -1)
        elif bad == "dtype":
            data = data.astype(np.float32)
        variable = tf.Variable(data, trainable=False)
        with pytest.raises(tf.errors.InvalidArgumentError, match="metadata|dtype|rank|vector|byte"):
            ops._OPS.dpointnet_csr_weight_grad(
                tf.ones((1, 1)), tf.ones((1, 4)), variable.handle, tf.ones((256, 4)),
                Tindex=index, n_post=1, n_edges=(2**62 if bad == "overflow" else 1),
                n_pairs=0, write_csr_weight_gradient=True, uint8_types=True,
            )
    finally:
        conn.close()


@pytest.mark.parametrize("traversal", ["scatter", "cooperative", "fanout", "gather"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
@pytest.mark.parametrize("batch", [1, 32])
def test_gpu_each_traversal_and_resource_identity_cache(monkeypatch, traversal, dtype, index, batch):
    native()
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *a: index)
    posts = 521
    width, edges = (2, 2201) if traversal == "fanout" else (7, 41)
    rng = np.random.default_rng(971)
    rows = np.arange(edges) % width
    endpoints = np.column_stack(((np.arange(edges) // width) % posts, rows))
    types = rng.choice(np.array([0, 255], np.int64), size=edges)
    basis = rng.uniform(-0.3, 0.8, (256, 4)).astype(dtype)
    weights = rng.uniform(-0.2, 0.3, edges).astype(np.float32)
    spikes = rng.integers(0, 4, (batch, width)).astype(dtype)
    conns = [
        ops.build_csr_connectivity(
            endpoints if i == 0 else np.column_stack(((endpoints[:, 0]+13) % posts, rows)),
            types, width, posts, 256, sort_by_target=True, host_lifecycle=True, uint8_types=True,
        ) for i in range(2)
    ]
    try:
        # One OpKernel instance must invalidate its cached CSC/tile tables on
        # actual allocation identity, not merely equal shape or source width.
        kwargs = dict(
            Tindex=index, n_post=posts, n_edges=[edges, edges], n_pairs=[0, 0],
            uint8_types=True, target_tile_size=256,
            cooperative_rows=[traversal == "cooperative"]*2, fanout_rows=traversal == "fanout",
            gather_sources=[traversal == "gather"]*2,
        )
        @tf.function
        def run(handle, w):
            return ops._OPS.dpointnet_combined_sweep_csr_master(
                [tf.constant(spikes)]*2, [w]*2, [handle]*2, tf.constant(basis), **kwargs,
            )
        for offset in (0, 1, 0):
            conn = conns[offset]
            w = CsrMasterLayout(conn, host_lifecycle=True).host_to_live(weights)
            actual = run(conn["metadata_handle"], tf.constant(w))
            expected = np.zeros((batch, posts, 4), np.float64)
            for sample in range(batch):
                products = (
                    spikes[sample, rows].astype(np.float64)[:, None]
                    * weights.astype(np.float64)[:, None] * basis[types].astype(np.float64)
                )
                np.add.at(expected[sample], (endpoints[:, 0]+13*offset) % posts, products)
            expected *= 2
            np.testing.assert_allclose(
                actual, expected.reshape(-1, 4),
                rtol=0.002 if dtype == np.float16 else 3e-5,
                atol=0.01 if dtype == np.float16 else 3e-5,
            )
    finally:
        for conn in conns:
            conn.close()


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("batch", [1, 3, 32])
def test_gpu_actual_cell_named_full_remainder_adam_and_canonical_export(monkeypatch, tmp_path, mixed, batch):
    native()
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet import acceleration
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner

    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_FP32_MASTERS", "1")
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_HOST_LIFECYCLE", "1")
    original_policy = tf.keras.mixed_precision.global_policy()
    def construct(typed):
        monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_CSR_UINT8_TYPES", str(int(typed)))
        network, inputs, arguments = make_cell(
            "nest", fused=True, two_inputs=True, return_spec=True, selective=mixed,
            train_recurrent=True, batch_size=batch, acceleration_profile="auto",
            _canonical_gradient_boundary=True, use_fused_recurrent_accumulation=(mixed and batch == 32),
            **({"use_combined_current_sweep": True} if batch == 3 else {}),
        )
        for props in inputs.values():
            props["input_type"] = "spikes"
        inputs["background"]["options"]["trainable"] = False
        arguments["_performance_auto_context"] = acceleration.performance_projection_context(
            network, inputs, dt=1., max_delay=5., standard_cell=True, single_replica=True,
        )
        cell = GLIF3Cell(network, inputs, **arguments)
        states = list(cell.zero_state(batch, cell.compute_dtype))
        states[1] = tf.tile(tf.constant([[1.1, 0.8]], states[1].dtype), [batch, 1])
        width = int(cell.inputs_idx[-1])
        sequence = tf.ones((batch, 7, width), cell.compute_dtype)
        sequence_input = tf.keras.Input(shape=(None, width), dtype=cell.compute_dtype)
        state_inputs = [tf.keras.Input(shape=x.shape[1:], dtype=x.dtype) for x in states]
        layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
        layer._autocast = False
        out = layer(sequence_input, initial_state=state_inputs)
        core = tf.keras.Model([sequence_input, *state_inputs], [*out[0], *out[1:]])
        masters = [cell.recurrent_weight_values, *[net["input_weight_values"] for net in cell.inputs.values()]]
        trainable = [w for w in masters if w.trainable]
        optimizer = tf.keras.optimizers.Adam(0.001)
        optimizer.build(trainable)
        runners = [None, *[SegmentedRecomputeRunner(core, 7, chunk, 2) for chunk in (3, 7)]]
        def evaluate(runner):
            with tf.GradientTape() as tape:
                values = core([sequence, *states]) if runner is None else runner(sequence, states)
                loss = tf.reduce_mean(tf.cast(values[0], tf.float32)) + tf.reduce_mean(values[1])
            return values, tape.gradient(loss, trainable)
        runs = [tf.function(lambda runner=r: evaluate(runner)) for r in runners]
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, noise_seed=cell.noise_seed, noise_stream=cell.noise_stream,
            **{f"master_{i}": w.value for i, w in enumerate(masters)},
        )
        return cell, masters, trainable, optimizer, runs, checkpoint

    try:
        baseline, candidate = construct(False), construct(True)
        a, b = baseline[0], candidate[0]
        assert a.acceleration_report["selected"] == b.acceleration_report["selected"]
        assert b.acceleration_report["experimental_csr_uint8_types"] == "csr_live_uint8_type_tail_v1"
        assert a.acceleration_report["experimental_csr_uint8_types"] == "disabled"
        assert b.acceleration_report["experimental_projection"] == "optimized_combined_csr_fp32_master"
        assert b.acceleration_report["experimental_csr_lifecycle"] == "host_numpy_uint32_v1"
        for name in ("<recurrent>", *b.inputs):
            np.testing.assert_array_equal(a.canonical_weight_values(name), b.canonical_weight_values(name))
            canonical = b.canonical_weight_values(name)
            np.testing.assert_array_equal(a.weight_loss_reference(canonical, name), b.weight_loss_reference(canonical, name))
            np.testing.assert_array_equal(a.weight_loss_indices(np.array([1, 0]), name), b.weight_loss_indices(np.array([1, 0]), name))
        def verify(item):
            values = [run() for run in item[4]]
            for got in values[1:]:
                for x, y in zip(tf.nest.flatten(got), tf.nest.flatten(values[0])):
                    assert x is not None and y is not None
                    np.testing.assert_allclose(x, y, rtol=3e-3, atol=2e-3)
            assert all(np.any(g.numpy() != 0) for g in values[0][1])
            return values[-1]
        def update(item):
            _, gradient = verify(item)
            item[3].apply_gradients(zip(gradient, item[2]))
            item[0].refresh_recurrent_weight_shadow()
        for item in (baseline, candidate):
            update(item)
        for x, y in zip(tf.nest.flatten(verify(baseline)), tf.nest.flatten(verify(candidate))):
            np.testing.assert_array_equal(x, y)
        graph = candidate[4][1].get_concrete_function().graph.as_graph_def()
        nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
        consumers = [n for n in nodes if n.op in (
            "DpointnetCsrSpikeForward", "DpointnetCsrWeightGrad", "DpointnetCsrSpikeGradCsrMaster",
            "DpointnetCsrSpikeGradAccumulateCsrMaster", "DpointnetCombinedSweepCsrMaster",
        )]
        assert consumers and all(n.attr["uint8_types"].b for n in consumers)
        assert not any(n.op in ("DpointnetCsrReorder", "DpointnetCsrRestore") for n in nodes)
        saved = candidate[5].save(str(tmp_path / "nonzero_adam"))
        expected_frozen = candidate[1][-1].numpy().copy()
        for item in (baseline, candidate):
            update(item)
        expected = [w.numpy().copy() for w in candidate[1]]
        slots = [v.numpy().copy() for v in candidate[3].variables]
        for w in candidate[1]:
            w.assign(tf.zeros_like(w))
        for v in candidate[3].variables:
            v.assign(tf.zeros_like(v))
        candidate[5].restore(saved).assert_consumed()
        assert int(candidate[3].iterations.numpy()) == 1
        assert any(np.any(v.numpy() != 0) for v in candidate[3]._momentums)
        update(candidate)
        for got, want in zip(candidate[1], expected):
            np.testing.assert_array_equal(got, want)
        for got, want in zip(candidate[3].variables, slots):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(candidate[1][-1], expected_frozen)
        for name in ("<recurrent>", *b.inputs):
            np.testing.assert_array_equal(a.canonical_weight_values(name), b.canonical_weight_values(name))
        for w in candidate[2]:
            assert np.all(np.where(w.constraint.condition.numpy(), w.numpy() >= 0, w.numpy() <= 0))
        from bmtk.simulator.dpointnet.weights import ModelWeights
        import pandas as pd
        exports = []
        for cell in (a, b):
            adapter = SimpleNamespace(
                inputs_populations=list(cell.inputs),
                get_network=lambda name: SimpleNamespace(
                    connection_table=pd.DataFrame({"canonical_row": [0, 1]}),
                    canonical_edge_order=None,
                ),
            )
            exports.append(ModelWeights(adapter, cell, deep_copy=False).to_dataframe())
        pd.testing.assert_frame_equal(exports[0], exports[1], check_exact=True)
        np.testing.assert_array_equal(exports[1]["canonical_row"], [0, 1, 0, 1, 0, 1])
        census = b.csr_host_lifecycle_metadata()
        for name, report in census.items():
            conn = b.recurrent_fused_connectivity if name == "<recurrent>" else b.inputs[name]["fused_connectivity"]
            alias = b._master_recurrent_connectivity if name == "<recurrent>" else b.inputs[name]["master_connectivity"]
            assert conn is alias
            raw = tf.raw_ops.ReadVariableOp(resource=conn["metadata_handle"], dtype=tf.as_dtype(conn["index_dtype"]))
            assert tuple(raw.shape) == tuple(report["metadata_shape"])
            assert raw.dtype.size * int(np.prod(raw.shape)) == report["native_metadata_bytes"]
            assert report["type_dtype"] == "uint8"
            assert report["type_logical_min"] == report["type_logical_max"] == 0
            specs = conn["metadata_resource_specs"]
            assert len(specs) == 1 and specs[0]["handle"] is conn["metadata_handle"]
    finally:
        tf.keras.mixed_precision.set_global_policy(original_policy)
