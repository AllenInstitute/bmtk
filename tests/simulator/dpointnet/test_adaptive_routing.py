"""Experimental adaptive routing (adaptive_segment_ticket_v1) for the CSR-master sweep.

Independent FP64 oracles for the forward and every affected per-source VJP, both
routing modes on the same source/binaries, registry/cache lifetime, topology
rejection, explicit capability/flag errors and actual-cell integration.
"""

import gc
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops
from bmtk.simulator.dpointnet.csr_master import CsrMasterLayout

BASE_FLAGS = (
    "_DPOINTNET_EXPERIMENTAL_CSR_FP32_MASTERS",
    "_DPOINTNET_EXPERIMENTAL_CSR_HOST_LIFECYCLE",
    "_DPOINTNET_EXPERIMENTAL_CSR_UINT8_TYPES",
)
N_TYPES = 6


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def native():
    if not tf.config.get_visible_devices("GPU") or not ops.combined_sweep_available():
        pytest.skip("native GPU test requires the isolated rebuilt library")
    assert ops.adaptive_routing_available()


# ---------------------------------------------------------------------------
# Host-only policy, flag and capability checks
# ---------------------------------------------------------------------------

def conn_stub(n_sources, n_edges, n_post, repeated):
    return {"n_sources": n_sources, "n_edges": n_edges, "n_post": n_post,
            "has_repeated_targets": repeated}


def test_policy_is_explicit_and_deterministic():
    recurrent = conn_stub(199974, 24450554, 66658, False)
    lgn = conn_stub(34800, 31633264, 66658, True)
    bkg = conn_stub(100, 266632, 66658, False)
    # Above the gather in-degree bound; unknown repetition / high fanout.
    unknown = conn_stub(10000, 2000000, 66658, None)
    high_unique = conn_stub(1000, 2000000, 66658, False)
    plan = ops.adaptive_routing_plan([recurrent, lgn, bkg, unknown, high_unique],
                                     target_tile_size=2048, gather_sources=True)
    assert plan["traversals"] == ["warp_batches", "block_rounds", "gather", "block_rounds",
                                  "block_rounds"]
    assert plan["codes"] == [1, 2, 3, 2, 2]
    assert plan["schema"] == ops.ADAPTIVE_ROUTING_SCHEMA == "adaptive_segment_ticket_v1"
    assert (plan["wide_tile"], plan["narrow_tile"]) == (2048, 512)
    no_gather = ops.adaptive_routing_plan([recurrent, lgn, bkg], gather_sources=False)
    assert no_gather["traversals"] == ["warp_batches", "block_rounds", "block_rounds"]
    assert plan == ops.adaptive_routing_plan([recurrent, lgn, bkg, unknown, high_unique],
                                             target_tile_size=2048, gather_sources=True)


def test_disabled_geometry_mirrors_native_dispatch():
    recurrent = conn_stub(199974, 24450554, 66658, False)
    lgn = conn_stub(34800, 31633264, 66658, True)
    bkg = conn_stub(100, 266632, 66658, False)
    short_repeat = conn_stub(100000, 20000000, 66658, True)
    auto = ops.combined_routing_geometry(
        [recurrent, lgn, bkg, short_repeat], target_tile_size=2048, cooperative_rows=True,
        fanout_rows=True, gather_sources=True,
    )
    assert [g["traversal"] for g in auto] == [
        "cooperative_rows", "fanout_rounds", "gather", "scatter_rows"]
    assert all(g == {"schema": "dense_tile_starts", "traversal": g["traversal"], "tile": 2048} for g in auto)
    plain = ops.combined_routing_geometry([recurrent, lgn, bkg], target_tile_size=1024)
    assert [g["traversal"] for g in plain] == ["scatter_rows", "balanced_rounds", "balanced_rounds"]
    adaptive = ops.combined_routing_geometry(
        [recurrent, lgn, bkg], target_tile_size=2048, gather_sources=True, adaptive_routing=True,
    )
    assert [(g["traversal"], g["wide_tile"], g["narrow_tile"]) for g in adaptive] == [
        ("warp_batches", 2048, 512), ("block_rounds", 2048, 512), ("gather", 2048, 512)]


@pytest.mark.parametrize("wide,narrow", [
    (256, 256), (512, 256), (1024, 256), (2048, 512), (3072, 512), (4096, 1024), (5120, 1024),
])
def test_narrow_tile_mapping(wide, narrow):
    assert ops.adaptive_narrow_tile(wide) == narrow
    assert ops.adaptive_routing_plan([], target_tile_size=wide)["narrow_tile"] == narrow


@pytest.mark.parametrize("missing", ["census", "adaptive_routing", "adaptive_traversal"])
def test_old_or_partial_library_reports_unavailable(monkeypatch, missing):
    parameters = {"uint8_types", "adaptive_routing", "adaptive_traversal", "target_tile_size",
                  "cooperative_rows"} - {missing}

    def make(names):
        def operator(**kwargs):
            return None
        operator.__signature__ = inspect.Signature([
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in sorted(names)
        ])
        return operator

    library = {
        "dpointnet_combined_sweep": make({"target_tile_size", "cooperative_rows"}),
        "dpointnet_combined_sweep_csr_master": make(parameters),
    }
    if missing != "census":
        library["dpointnet_routing_census"] = make(set())
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(**library))
    monkeypatch.setattr(ops, "fused_cuda_available", lambda: True)
    assert not ops.adaptive_routing_available()
    with pytest.raises(RuntimeError, match="Rebuild both CUDA operators for experimental adaptive"):
        ops.require_adaptive_routing()
    with pytest.raises(RuntimeError, match="Rebuild both CUDA operators for experimental adaptive"):
        ops.fused_combined_currents([], [], [], None, [], csr_master_weights=True,
                                    adaptive_routing=True)


@pytest.mark.parametrize("value", [None, 0, 1, "1"])
def test_low_level_request_requires_boolean(value):
    with pytest.raises(ValueError, match="adaptive_routing must be true or false"):
        ops.fused_combined_currents([], [], [], None, [], adaptive_routing=value)


def test_low_level_request_requires_csr_masters():
    with pytest.raises(ValueError, match="requires direct CSR FP32 master"):
        ops.fused_combined_currents([], [], [], None, [], adaptive_routing=True)


@pytest.mark.parametrize("value", ["auto", "true", "2", " 1"])
def test_invalid_flag_value_rejects(monkeypatch, value):
    from test_precision_credit import make_cell
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_ADAPTIVE_ROUTING", value)
    with pytest.raises(ValueError, match="ADAPTIVE_ROUTING must be 0 or 1"):
        make_cell()


@pytest.mark.parametrize("flags", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1)])
def test_unsupported_flag_compositions_reject(monkeypatch, flags):
    from test_precision_credit import make_cell
    for name, value in zip(BASE_FLAGS, flags):
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_ADAPTIVE_ROUTING", "1")
    with pytest.raises(ValueError, match="masters|lifecycle|uint8"):
        make_cell()


def test_cpu_composition_rejects(monkeypatch):
    from test_precision_credit import make_cell
    for name in BASE_FLAGS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_ADAPTIVE_ROUTING", "1")
    with pytest.raises(ValueError, match="native fused CUDA"):
        make_cell("nest")


# ---------------------------------------------------------------------------
# Native fixtures and FP64 oracle
# ---------------------------------------------------------------------------

def make_source(kind, rng, n_post, batch, dtype):
    """Return (indices[post, pre], types, weights, spikes) for one source family."""
    if kind == "lgn":
        # High fanout with repeated targets and exact duplicate edges.
        width, per_row = 6, 700
        pre = np.repeat(np.arange(width), per_row)
        post = rng.integers(n_post, size=pre.size)
        post[1::7] = post[0::7][: post[1::7].size]
        types = rng.integers(N_TYPES, size=pre.size)
        types[1::7] = types[0::7][: types[1::7].size]
        density = 0.7
    elif kind == "recurrent":
        # Short target-unique rows, several empty rows.
        width = 300
        lengths = rng.integers(0, 40, size=width)
        lengths[:9] = 0
        lengths[-1] = 17
        pre = np.repeat(np.arange(width), lengths)
        post = np.concatenate([rng.choice(n_post, size=n, replace=False) for n in lengths])
        types = rng.integers(N_TYPES, size=pre.size)
        density = 0.15
    elif kind == "bkg":
        # Few rows, four incoming edges per target.
        width = 4
        post = np.tile(np.arange(n_post), 4)
        pre = np.repeat(np.arange(width), n_post)
        types = rng.integers(N_TYPES, size=pre.size)
        density = 0.6
    elif kind == "repeat_low":
        width = 40
        pre = np.repeat(np.arange(width), 25)
        post = rng.integers(min(n_post, 30), size=pre.size)
        types = rng.integers(N_TYPES, size=pre.size)
        density = 0.4
    elif kind == "empty":
        width = 5
        pre = np.empty(0, np.int64)
        post = np.empty(0, np.int64)
        types = np.empty(0, np.int64)
        density = 0.5
    else:
        raise AssertionError(kind)
    weights = rng.uniform(-1.0, 1.0, size=pre.size).astype(np.float32)
    counts = rng.integers(1, 4, size=(batch, width))
    spikes = np.where(rng.random((batch, width)) < density, counts, 0).astype(dtype)
    if width > 1:
        spikes[:, -1] = 0  # connected row with no activity keeps spike credit
    return np.column_stack((post, pre)).astype(np.int64), types.astype(np.int64), weights, spikes


def amax(values):
    values = np.asarray(values, np.float64)
    return float(np.max(np.abs(values))) if values.size else 0.0


def fp64_oracle(sources, basis, dy):
    batch, n_post, _ = dy.shape
    out = np.zeros((batch, n_post, 4), np.float64)
    dzs, dws = [], []
    basis64 = basis.astype(np.float64)
    dy64 = dy.astype(np.float64)
    for indices, types, weights, spikes in sources:
        post, pre = indices[:, 0], indices[:, 1]
        z = spikes.astype(np.float64)
        w = weights.astype(np.float64)
        contribution = z[:, pre, None] * (w[:, None] * basis64[types])[None]
        for b in range(batch):
            np.add.at(out[b], post, contribution[b])
        projected = np.einsum("bek,ek->be", dy64[:, post], basis64[types])
        dz = np.zeros_like(z)
        for b in range(batch):
            np.add.at(dz[b], pre, projected[b] * w)
        dzs.append(dz)
        dws.append(np.sum(z[:, pre] * projected, axis=0))
    return out.reshape(batch * n_post, 4), dzs, dws


def build(indices, types, width, n_post):
    return ops.build_csr_connectivity(
        indices, types, width, n_post, N_TYPES, build_compact_pairs=True, sort_by_target=True,
        host_lifecycle=True, uint8_types=True,
    )


def run_sweep(sources, conns, basis, dy, *, adaptive, tile, gather, n_post, eager=False):
    layouts = [CsrMasterLayout(conn, host_lifecycle=True) for conn in conns]
    zs = [tf.Variable(source[3]) for source in sources]
    ws = [tf.Variable(layout.host_to_live(source[2])) for layout, source in zip(layouts, sources)]
    coefficients = tf.constant(basis)

    def evaluate():
        with tf.GradientTape() as tape:
            proxies = [
                ops.fused_spike_currents(
                    z, w, w, conn, coefficients, n_post, True, vjp_only=True,
                    csr_master_weights=True, use_packed_sm120_backward=False,
                    write_csr_weight_gradient=True,
                ) for z, w, conn in zip(zs, ws, conns)
            ]
            out = ops.fused_combined_currents(
                zs, ws, conns, coefficients, proxies, target_tile_size=tile,
                csr_master_weights=True, gather_sources=gather, fanout_rows=True,
                cooperative_rows=True, adaptive_routing=adaptive,
            )
            loss = tf.reduce_sum(tf.cast(out, tf.float32) * dy.reshape(-1, 4))
        return out, tape.gradient(loss, zs + ws)

    run = evaluate if eager else tf.function(evaluate)
    out, grads = run()
    return out.numpy(), [g.numpy() for g in grads], layouts, run


def sweep_nodes(run):
    graph = run.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
    return [n for n in nodes if n.op == "DpointnetCombinedSweepCsrMaster"]


def check_against_oracle(out, grads, layouts, reference, dtype):
    expected, dzs, dws = reference
    scale = max(1.0, float(np.max(np.abs(expected))))
    tol = 4e-3 if dtype == np.float16 else 2e-6
    np.testing.assert_allclose(out.astype(np.float64), expected, rtol=tol, atol=tol * scale)
    relative = np.linalg.norm(out.astype(np.float64) - expected) / max(np.linalg.norm(expected), 1e-30)
    assert relative < (1e-3 if dtype == np.float16 else 1e-6)
    count = len(dzs)
    for got, want in zip(grads[:count], dzs):
        gscale = max(1.0, float(np.max(np.abs(want))) if want.size else 1.0)
        np.testing.assert_allclose(got, want, rtol=0.02 if dtype == np.float16 else 2e-5,
                                   atol=(0.02 if dtype == np.float16 else 2e-5) * gscale)
    for got, want, layout in zip(grads[count:], dws, layouts):
        gscale = max(1.0, float(np.max(np.abs(want))) if want.size else 1.0)
        np.testing.assert_allclose(got, layout.host_to_live(want),
                                   rtol=0.02 if dtype == np.float16 else 2e-5,
                                   atol=(0.02 if dtype == np.float16 else 2e-5) * gscale)


GEOMETRIES = {
    # n_post, wide tile: masks (<=64 tiles), binary search (>64 tiles), dynamic shared.
    "masks": (4099, 2048),
    "search": (33000, 256),
    "wide5120": (10243, 5120),
}


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [1, 3, 32])
@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
@pytest.mark.parametrize("gather", [False, True])
def test_gpu_fp64_forward_and_vjps_both_modes(monkeypatch, geometry, dtype, batch, index, gather):
    native()
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *a: index)
    # Only the four-row BKG-like source is gather-eligible, as in the full model.
    monkeypatch.setattr(ops, "combined_gather_eligible", lambda conn: conn["n_sources"] == 4)
    n_post, tile = GEOMETRIES[geometry]
    rng = np.random.default_rng(2026 + batch)
    kinds = ("recurrent", "lgn", "bkg", "repeat_low", "empty")
    sources = [make_source(kind, rng, n_post, batch, dtype) for kind in kinds]
    sources[3][3][:] = 0  # an entirely silent source still receives spike credit
    basis = rng.uniform(-0.8, 1.1, (N_TYPES, 4)).astype(dtype)
    dy = rng.uniform(-0.5, 0.5, (batch, n_post, 4)).astype(dtype)
    reference = fp64_oracle(sources, basis, dy)
    conns = [build(s[0], s[1], s[3].shape[1], n_post) for s in sources]
    try:
        results = {}
        for adaptive in (False, True):
            out, grads, layouts, run = run_sweep(
                sources, conns, basis, dy, adaptive=adaptive, tile=tile, gather=gather,
                n_post=n_post,
            )
            check_against_oracle(out, grads, layouts, reference, dtype)
            nodes = sweep_nodes(run)
            assert len(nodes) == 1
            assert nodes[0].attr["adaptive_routing"].b is adaptive
            assert nodes[0].attr["uint8_types"].b
            assert nodes[0].attr["target_tile_size"].i == tile
            if adaptive:
                plan = ops.adaptive_routing_plan(conns, target_tile_size=tile, gather_sources=gather)
                assert list(nodes[0].attr["adaptive_traversal"].list.i) == plan["codes"]
                assert plan["traversals"][:2] == ["warp_batches", "block_rounds"]
                assert plan["traversals"][2] == ("gather" if gather else "block_rounds")
            results[adaptive] = (out, grads)
            assert np.any(grads[3] != 0), "silent source lost spike credit"
            assert np.any(grads[0][:, -1] != 0), "silent connected row lost spike credit"
        a, b = results[False], results[True]
        np.testing.assert_allclose(
            b[0].astype(np.float64), a[0].astype(np.float64),
            rtol=4e-3 if dtype == np.float16 else 2e-6,
            atol=(4e-3 if dtype == np.float16 else 2e-6) * max(1.0, amax(a[0])),
        )
        for x, y in zip(a[1], b[1]):
            # Per-source VJPs keep existing routing; only atomic order may differ.
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5 * max(1.0, amax(x)))
    finally:
        for conn in conns:
            conn.close()


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_eager_execution_matches_oracle(monkeypatch, dtype):
    native()
    n_post = 4099
    rng = np.random.default_rng(5)
    sources = [make_source(kind, rng, n_post, 3, dtype) for kind in ("recurrent", "lgn")]
    basis = rng.uniform(-0.8, 1.1, (N_TYPES, 4)).astype(dtype)
    dy = rng.uniform(-0.5, 0.5, (3, n_post, 4)).astype(dtype)
    conns = [build(s[0], s[1], s[3].shape[1], n_post) for s in sources]
    try:
        out, grads, layouts, _ = run_sweep(sources, conns, basis, dy, adaptive=True, tile=2048,
                                           gather=False, n_post=n_post, eager=True)
        check_against_oracle(out, grads, layouts, fp64_oracle(sources, basis, dy), dtype)
    finally:
        for conn in conns:
            conn.close()


def profiled_kernels(trace_dir):
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    from pathlib import Path
    found = set()
    for path in Path(trace_dir).rglob("*.xplane.pb"):
        space = xplane_pb2.XSpace()
        space.ParseFromString(path.read_bytes())
        for plane in space.planes:
            if not plane.name.startswith("/device:GPU:"):
                continue
            for line in plane.lines:
                for event in line.events:
                    op = ""
                    for stat in event.stats:
                        if plane.stat_metadata[stat.metadata_id].name == "tf_op":
                            field = stat.WhichOneof("value")
                            value = getattr(stat, field)
                            op = plane.stat_metadata[value].name if field == "ref_value" else str(value)
                    found.add((op.split(":")[-1], plane.event_metadata[event.metadata_id].name))
    return found


@pytest.mark.parametrize("adaptive", [False, True])
def test_gpu_profiler_kernel_names_keep_parent_contract(tmp_path, adaptive):
    native()
    n_post = 4099
    rng = np.random.default_rng(3)
    sources = [make_source(kind, rng, n_post, 32, np.float16) for kind in ("recurrent", "lgn", "bkg")]
    basis = rng.uniform(-0.8, 1.1, (N_TYPES, 4)).astype(np.float16)
    dy = rng.uniform(-0.5, 0.5, (32, n_post, 4)).astype(np.float16)
    conns = [build(s[0], s[1], s[3].shape[1], n_post) for s in sources]
    try:
        _, _, _, run = run_sweep(sources, conns, basis, dy, adaptive=adaptive, tile=2048,
                                 gather=True, n_post=n_post)
        tf.profiler.experimental.start(str(tmp_path))
        try:
            run()
        finally:
            tf.profiler.experimental.stop()
        kernels = profiled_kernels(tmp_path)
        forward = {name for op, name in kernels if op == "DpointnetCombinedSweepCsrMaster"}
        assert any("DpointnetCombinedSweepKernel" in name for name in forward), forward
        assert any("adaptive_routing::" in name for name in forward) is adaptive
        vjp = {name for op, name in kernels
               if op in ("DpointnetCsrSpikeGradCsrMaster", "DpointnetCsrSpikeGradAccumulateCsrMaster")}
        assert any("SpikeGrad" in name for name in vjp), vjp
        assert not any(op == "DpointnetRoutingCensus" for op, _ in kernels)
    finally:
        for conn in conns:
            conn.close()


def census(conns):
    return ops.combined_routing_census(conns)


def raw_sweep(conns, spikes, weights, basis, *, tile, codes):
    return ops._OPS.dpointnet_combined_sweep_csr_master(
        spikes, weights, [c["metadata_handle"] for c in conns], basis,
        Tindex=tf.as_dtype(conns[0]["index_dtype"]), n_post=conns[0]["n_post"],
        n_edges=[c["n_edges"] for c in conns], n_pairs=[c["n_pairs"] for c in conns],
        target_tile_size=tile, uint8_types=True, adaptive_routing=True, adaptive_traversal=codes,
    )


def dense_reference(indices, types, weights, spikes, basis, n_post):
    batch = spikes.shape[0]
    out = np.zeros((batch, n_post, 4), np.float64)
    contribution = spikes.astype(np.float64)[:, indices[:, 1], None] * (
        weights.astype(np.float64)[:, None] * basis.astype(np.float64)[types])[None]
    for b in range(batch):
        np.add.at(out[b], indices[:, 0], contribution[b])
    return out.reshape(-1, 4)


@pytest.mark.parametrize("traversal", ["warp_batches", "block_rounds", "gather"])
@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
def test_gpu_identity_change_refreshes_shared_tables(monkeypatch, traversal, index):
    native()
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *a: index)
    n_post, width, edges = 4099, 64, 64 * 12
    rng = np.random.default_rng(31)
    rows = np.arange(edges) % width
    endpoints = np.column_stack(((np.arange(edges) * 7) % n_post, rows))
    types = rng.integers(N_TYPES, size=edges)
    weights = rng.uniform(-0.5, 0.5, edges).astype(np.float32)
    spikes = rng.integers(0, 3, (2, width)).astype(np.float32)
    basis = rng.uniform(-0.3, 0.8, (N_TYPES, 4)).astype(np.float32)
    shifted = np.column_stack(((endpoints[:, 0] + 13) % n_post, rows))
    conns = [build(e, types, width, n_post) for e in (endpoints, shifted)]
    code = ops.ADAPTIVE_TRAVERSAL_CODES[traversal]
    try:
        @tf.function
        def run(handle, w):
            return ops._OPS.dpointnet_combined_sweep_csr_master(
                [tf.constant(spikes)] * 2, [w] * 2, [handle] * 2, tf.constant(basis),
                Tindex=index, n_post=n_post, n_edges=[edges] * 2, n_pairs=[conns[0]["n_pairs"]] * 2,
                target_tile_size=2048, uint8_types=True, adaptive_routing=True,
                adaptive_traversal=[code] * 2,
            )
        for which in (0, 1, 0, 1):
            conn = conns[which]
            live = CsrMasterLayout(conn, host_lifecycle=True).host_to_live(weights)
            actual = run(conn["metadata_handle"], tf.constant(live)).numpy()
            expected = 2 * dense_reference((endpoints, shifted)[which], types, weights, spikes,
                                           basis, n_post)
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        # The operator now holds only the last connectivity's shared table; the
        # previous identity's table was released with its last reference.
        released, held = census(conns)
        assert released["routing_table_bytes"] == 0
        if traversal == "gather":
            assert held["gather_tables"] == 1 and held["segment_tables"] == 0
        else:
            assert held["segment_tables"] == 1 and held["gather_tables"] == 0
        assert held["dense_tile_bytes"] == 0 and held["routing_table_bytes"] > 0
        assert held["traversal"] == traversal
    finally:
        for conn in conns:
            conn.close()


def test_gpu_topology_replacement_rejects():
    native()
    n_post = 4099
    conn = build(np.array([[2047, 0], [5, 1]]), np.array([1, 2]), 2, n_post)
    try:
        spikes = tf.ones((1, 2))
        w = tf.constant(CsrMasterLayout(conn, host_lifecycle=True).host_to_live(
            np.array([1.0, -2.0], np.float32)))

        @tf.function
        def run():
            return raw_sweep([conn, conn], [spikes] * 2, [w] * 2, tf.ones((N_TYPES, 4)),
                             tile=2048, codes=[1, 2])

        before = run().numpy()
        assert before[2047, 0] == pytest.approx(2.0) and before[5, 0] == pytest.approx(-4.0)
        handle = conn["metadata_handle"]
        metadata = tf.raw_ops.ReadVariableOp(resource=handle, dtype=tf.uint32)
        changed = tf.tensor_scatter_nd_update(metadata, [[0]], tf.constant([2048], tf.uint32))
        tf.raw_ops.AssignVariableOp(resource=handle, value=changed)
        with pytest.raises(tf.errors.InvalidArgumentError, match="immutable connectivity"):
            run().numpy()
    finally:
        conn.close()


def test_gpu_live_weights_are_read_each_launch():
    native()
    conn = build(np.array([[0, 0], [4098, 1]]), np.array([0, 1]), 2, 4099)
    try:
        layout = CsrMasterLayout(conn, host_lifecycle=True)
        w = tf.Variable(layout.host_to_live(np.array([1.0, 2.0], np.float32)))

        @tf.function
        def run():
            return raw_sweep([conn, conn], [tf.ones((1, 2))] * 2, [w] * 2, tf.ones((N_TYPES, 4)),
                             tile=2048, codes=[1, 2])

        np.testing.assert_array_equal(run().numpy()[[0, 4098], 0], [2.0, 4.0])
        w.assign(layout.host_to_live(np.array([5.0, -2.0], np.float32)))
        np.testing.assert_array_equal(run().numpy()[[0, 4098], 0], [10.0, -4.0])
    finally:
        conn.close()


@pytest.mark.parametrize("bad", ["length", "code", "no_uint8", "not_adaptive"])
def test_gpu_native_rejects_invalid_adaptive_attributes(bad):
    native()
    conn = build(np.array([[0, 0]]), np.array([1]), 1, 8)
    try:
        w = tf.ones((1,))
        kwargs = dict(
            Tindex=tf.uint32, n_post=8, n_edges=[1, 1], n_pairs=[conn["n_pairs"]] * 2,
            target_tile_size=2048, uint8_types=True, adaptive_routing=True,
            adaptive_traversal=[1, 2],
        )
        if bad == "length":
            kwargs["adaptive_traversal"] = [1]
        elif bad == "code":
            kwargs["adaptive_traversal"] = [1, 4]
        elif bad == "no_uint8":
            kwargs["uint8_types"] = False
        else:
            kwargs["adaptive_routing"] = False
        with pytest.raises((tf.errors.InvalidArgumentError, ValueError),
                           match="adaptive|uint8|traversal"):
            ops._OPS.dpointnet_combined_sweep_csr_master(
                [tf.ones((1, 1))] * 2, [w] * 2, [conn["metadata_handle"]] * 2, tf.ones((N_TYPES, 4)),
                **kwargs,
            )
    finally:
        conn.close()


def test_gpu_rejects_non_uint8_connectivity():
    native()
    conn = ops.build_csr_connectivity([[0, 0]], [1], 1, 8, N_TYPES, build_compact_pairs=True,
                                      sort_by_target=True, host_lifecycle=True)
    try:
        with pytest.raises(ValueError, match="requires uint8"):
            ops.fused_combined_currents(
                [tf.ones((1, 1))] * 2, [tf.ones((1,))] * 2, [conn, conn], tf.ones((N_TYPES, 4)),
                [tf.zeros((8, 4))] * 2, csr_master_weights=True, adaptive_routing=True,
            )
    finally:
        conn.close()


def test_gpu_registry_sharing_bounded_shapes_and_release():
    native()
    n_post = 4099
    rng = np.random.default_rng(77)
    indices, types, weights, spikes = make_source("recurrent", rng, n_post, 32, np.float32)
    lgn = make_source("lgn", rng, n_post, 32, np.float32)
    conns = [build(indices, types, spikes.shape[1], n_post), build(lgn[0], lgn[1], 6, n_post)]
    try:
        live = [CsrMasterLayout(c, host_lifecycle=True).host_to_live(w)
                for c, w in zip(conns, (weights, lgn[2]))]
        basis = tf.constant(rng.uniform(-1, 1, (N_TYPES, 4)).astype(np.float32))
        signature = [tf.TensorSpec((None, spikes.shape[1]), tf.float32),
                     tf.TensorSpec((None, 6), tf.float32)]

        def make(adaptive):
            def body(z0, z1):
                if adaptive:
                    return raw_sweep(conns, [z0, z1], [tf.constant(x) for x in live], basis,
                                     tile=2048, codes=[1, 2])
                return ops._OPS.dpointnet_combined_sweep_csr_master(
                    [z0, z1], [tf.constant(x) for x in live], [c["metadata_handle"] for c in conns],
                    basis, Tindex=tf.uint32, n_post=n_post, n_edges=[c["n_edges"] for c in conns],
                    n_pairs=[c["n_pairs"] for c in conns], target_tile_size=2048, uint8_types=True,
                )
            return tf.function(body, input_signature=signature)

        first, second, baseline = make(True), make(True), make(False)
        for function in (first, second):
            for batch in (1, 32, 1):
                expected = dense_reference(indices, types, weights, spikes[:batch], basis.numpy(), n_post)
                expected += dense_reference(lgn[0], lgn[1], lgn[2], lgn[3][:batch], basis.numpy(), n_post)
                actual = function(tf.constant(spikes[:batch]), tf.constant(lgn[3][:batch])).numpy()
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5 * np.abs(expected).max())
        items = census(conns)
        for item in items:
            # Two operators share one table per shape: B1 narrow 512, B32 wide 2048.
            assert item["segment_tables"] == 2
            assert item["built_tiles"] == [512, 2048]
            assert item["operators"] == 2
            assert item["dense_tile_bytes"] == 0 and item["dense_tile_tables"] == 0
            assert item["routing_table_bytes"] == item["segment_bytes"] + item["mask_bytes"] > 0
            assert item["transient_queue_bytes"] > 0
        assert [item["traversal"] for item in items] == ["warp_batches", "block_rounds"]
        baseline(tf.constant(spikes), tf.constant(lgn[3]))
        items = census(conns)
        for item in items:
            assert item["dense_tile_tables"] == 1 and item["dense_tile_bytes"] > 0
            assert item["operators"] == 3
        # Operator kernels (and their strong table references) live until TensorFlow
        # destroys them: delete the functions and clear the eager kernel cache.
        del first, second, baseline
        gc.collect()
        from tensorflow.python.eager import context
        context.context().clear_kernel_cache()
        gc.collect()
        released = census(conns)
        for item in released:
            assert item["segment_tables"] == 0 and item["routing_table_bytes"] == 0
    finally:
        for conn in conns:
            conn.close()


def test_gpu_segment_census_matches_independent_count():
    native()
    n_post = 4099
    rng = np.random.default_rng(9)
    indices, types, weights, spikes = make_source("recurrent", rng, n_post, 32, np.float32)
    conn = build(indices, types, spikes.shape[1], n_post)
    try:
        live = tf.constant(CsrMasterLayout(conn, host_lifecycle=True).host_to_live(weights))

        @tf.function
        def run():
            return raw_sweep([conn, conn], [tf.constant(spikes)] * 2, [live] * 2,
                             tf.ones((N_TYPES, 4)), tile=2048, codes=[1, 1])

        run()
        item = census([conn])[0]
        order = np.lexsort((indices[:, 0], indices[:, 1]))
        pre, post = indices[order, 1], indices[order, 0]
        tiles = post // 2048
        segments = int(np.count_nonzero(np.r_[True, (pre[1:] != pre[:-1]) | (tiles[1:] != tiles[:-1])])) if pre.size else 0
        width = spikes.shape[1]
        assert item["segments"] == segments
        # splits[width + 1] + starts[S + 1] uint32 words; masks one uint64 per row.
        assert item["segment_bytes"] == 4 * (width + 1) + 4 * (segments + 1)
        assert item["mask_bytes"] == 8 * width
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Actual cell: both modes, full/remainder replay, Adam, frozen named weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("batch", [1, 3, 32])
def test_gpu_actual_cell_both_modes_replay_adam_and_report(monkeypatch, mixed, batch):
    native()
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet import acceleration
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner

    for name in BASE_FLAGS:
        monkeypatch.setenv(name, "1")

    def construct(adaptive):
        monkeypatch.setenv("_DPOINTNET_EXPERIMENTAL_ADAPTIVE_ROUTING", str(int(adaptive)))
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
        return cell, masters, trainable, optimizer, runs

    def verify(item):
        values = [run() for run in item[4]]
        for got in values[1:]:
            for x, y in zip(tf.nest.flatten(got), tf.nest.flatten(values[0])):
                np.testing.assert_allclose(x, y, rtol=3e-3, atol=2e-3)
        assert all(np.any(g.numpy() != 0) for g in values[0][1])
        return values[-1]

    def update(item):
        _, gradient = verify(item)
        item[3].apply_gradients(zip(gradient, item[2]))
        item[0].refresh_recurrent_weight_shadow()

    try:
        baseline, candidate = construct(False), construct(True)
        a, b = baseline[0], candidate[0]
        assert a.acceleration_report["selected"] == b.acceleration_report["selected"]
        assert a.acceleration_report["experimental_adaptive_routing"] == "disabled"
        assert b.acceleration_report["experimental_adaptive_routing"] == "adaptive_segment_ticket_v1"
        assert b.acceleration_report["experimental_csr_uint8_types"] == "csr_live_uint8_type_tail_v1"
        assert b._use_combined_current_sweep
        assert b.acceleration_report["experimental_adaptive_routing_plan"] == b.adaptive_routing_plan()
        assert "experimental_adaptive_routing_plan" not in a.acceleration_report
        assert a.acceleration_report["experimental_routing_geometry"] == a.combined_routing_geometry()
        assert b.acceleration_report["experimental_routing_geometry"] == b.combined_routing_geometry()
        for name, item in b.combined_routing_geometry().items():
            assert item["schema"] == "adaptive_segment_ticket_v1"
            assert item["traversal"] == b.adaptive_routing_plan()["traversals"][name]
        for item in a.combined_routing_geometry().values():
            assert item["schema"] == "dense_tile_starts" and "tile" in item
        for x, y in zip(tf.nest.flatten(verify(baseline)), tf.nest.flatten(verify(candidate))):
            np.testing.assert_allclose(x, y, rtol=3e-3, atol=2e-3)
        frozen = candidate[1][-1].numpy().copy()
        for item in (baseline, candidate):
            update(item)
        for x, y in zip(tf.nest.flatten(verify(baseline)), tf.nest.flatten(verify(candidate))):
            np.testing.assert_allclose(x, y, rtol=3e-3, atol=2e-3)
        np.testing.assert_array_equal(candidate[1][-1].numpy(), frozen)
        for w in candidate[2]:
            assert np.all(np.where(w.constraint.condition.numpy(), w.numpy() >= 0, w.numpy() <= 0))
        for function, adaptive in ((baseline[4][1], False), (candidate[4][1], True)):
            graph = function.get_concrete_function().graph.as_graph_def()
            nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
            sweeps = [n for n in nodes if n.op == "DpointnetCombinedSweepCsrMaster"]
            assert sweeps and all(n.attr["adaptive_routing"].b is adaptive for n in sweeps)
            assert not any(n.op in ("DpointnetCsrReorder", "DpointnetCsrRestore") for n in nodes)
        report_a, report_b = a.adaptive_routing_report(), b.adaptive_routing_report()
        assert list(report_b) == ["recurrent", *b.inputs] == list(report_a)
        plan = b.adaptive_routing_plan()
        for name, item in report_b.items():
            assert item["adaptive"] is True and item["schema"] == "adaptive_segment_ticket_v1"
            assert item["policy_traversal"] == plan["traversals"][name] == item["traversal"]
            assert item["dense_tile_bytes"] == 0 and item["routing_table_bytes"] > 0
            assert item["last_selected_tile"] in (item["wide_tile"], item["narrow_tile"])
        for item in report_a.values():
            assert item["adaptive"] is False and item["segment_bytes"] == 0
            assert item["policy_traversal"] == item["traversal"]
            assert item["dense_tile_bytes"] > 0
            assert item["routing_table_bytes"] == item["dense_tile_bytes"] + item["gather_table_bytes"]
        for name in ("<recurrent>", *b.inputs):
            assert a.canonical_weight_values(name).shape == b.canonical_weight_values(name).shape
    finally:
        for item in ("baseline", "candidate"):
            cell = locals().get(item)
            if cell is not None:
                cell[0].close_fused_cuda()
