"""Locality row schedules for the FP16 Javier-derived recurrent accumulate backward."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops
from bmtk.simulator.dpointnet.row_schedule import (
    hilbert_index, locality_row_schedule, presynaptic_classes,
)


def _network(n=60, slabs=3, seed=3):
    """Distance-dependent edges ``[target, neuron + n * slab]`` with class-specific types."""
    rng = np.random.default_rng(seed)
    position = rng.uniform(0, 100, (n, 2))
    kind = rng.integers(0, 3, n)
    target, source, types = [], [], []
    for pre in range(n):
        distance = np.linalg.norm(position - position[pre], axis=1)
        posts = np.flatnonzero((distance < 35) & (np.arange(n) != pre))
        posts = rng.choice(posts, size=min(posts.size, 8), replace=False)
        for post in posts:
            target.append(post)
            source.append(pre + n * rng.integers(0, slabs))
            types.append(2 * kind[pre] + (kind[post] > 0))
    return np.column_stack((target, source)), np.array(types), position, kind, n * slabs


def test_hilbert_index_is_a_bijection_with_unit_steps():
    side = 16
    a, b = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    index = hilbert_index(a.ravel(), b.ravel(), bits=4)
    assert np.array_equal(np.sort(index), np.arange(side * side))
    order = np.argsort(index)
    steps = np.abs(np.diff(a.ravel()[order])) + np.abs(np.diff(b.ravel()[order]))
    assert np.all(steps == 1)


@pytest.mark.parametrize("use_positions", [True, False])
def test_schedule_is_a_slab_major_class_grouped_permutation(use_positions):
    indices, types, position, _, n_sources = _network()
    n = position.shape[0]
    schedule, report = locality_row_schedule(
        indices, types, n, n_sources, positions=position if use_positions else None,
    )
    assert schedule.dtype == np.uint32 and np.array_equal(np.sort(schedule), np.arange(n_sources))
    slabs = schedule.reshape(n_sources // n, n).astype(np.int64)
    assert np.array_equal(slabs // n, np.repeat(np.arange(n_sources // n), n).reshape(slabs.shape))
    assert np.all(slabs % n == slabs[0] % n)
    classes = presynaptic_classes(indices[:, 1] % n, types, n)[slabs[0]]
    assert np.all(np.diff(classes) >= 0)
    assert ("positions" in report) == use_positions and "classes" in report


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_spectral_schedule_handles_tiny_networks(n):
    """Test-sized networks (fewer neurons than three singular vectors) still schedule."""
    edges = np.array([[(i + 1) % n, i + n * (i % 2)] for i in range(n)])
    schedule, report = locality_row_schedule(edges, np.zeros(n, np.int64), n, 2 * n)
    assert np.array_equal(np.sort(schedule), np.arange(2 * n)) and "spectral" in report


def test_schedule_rejects_partial_slabs_and_bad_positions():
    indices, types, position, _, n_sources = _network()
    n = position.shape[0]
    with pytest.raises(ValueError, match="delay slabs"):
        locality_row_schedule(indices, types, n, n_sources + 1, positions=position)
    with pytest.raises(ValueError, match="positions"):
        locality_row_schedule(indices, types, n, n_sources, positions=position[:, :1])
    bad = position.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="positions"):
        locality_row_schedule(indices, types, n, n_sources, positions=bad)


@pytest.mark.parametrize(
    "schedule, match",
    [
        (np.arange(5), "permutation"),
        (np.r_[np.arange(5), 0], "permutation"),
        (np.r_[np.arange(5), 6], "permutation"),
        (np.arange(6).reshape(2, 3), "permutation"),
        (np.arange(6, dtype=np.float32), "permutation"),
    ],
)
def test_attach_rejects_invalid_schedules(schedule, match):
    conn = ops.build_csr_connectivity([[0, 0], [1, 5]], [0, 1], 6, 2, 2)
    try:
        with pytest.raises(ValueError, match=match):
            conn.attach_row_schedule(schedule)
        assert conn["row_schedule_handle"] is None
    finally:
        conn.close()


def test_attach_requires_uint32_metadata(monkeypatch):
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: tf.int64)
    conn = ops.build_csr_connectivity([[0, 0]], [0], 2, 1, 1)
    try:
        with pytest.raises(ValueError, match="uint32"):
            conn.attach_row_schedule(np.array([1, 0]))
    finally:
        conn.close()


def test_schedule_resource_is_owned_outside_the_metadata_census():
    conn = ops.build_csr_connectivity([[0, 0], [1, 2]], [0, 1], 3, 2, 2)
    try:
        conn.attach_row_schedule(np.array([2, 0, 1]), report="test")
        with pytest.raises(ValueError, match="already attached"):
            conn.attach_row_schedule(np.array([0, 1, 2]))
        assert conn["row_schedule_bytes"] == 12 and conn["row_schedule_report"] == "test"
        assert conn["row_schedule_handle"] is not None
        assert conn.metadata_resource_handles == (conn["metadata_handle"],)
        assert len(conn.metadata_resource_specs) == 1
        values = tf.raw_ops.ReadVariableOp(resource=conn["row_schedule_handle"], dtype=tf.uint32)
        assert np.array_equal(values.numpy(), [2, 0, 1])
    finally:
        conn.close()
    plain = ops.build_csr_connectivity([[0, 0]], [0], 1, 1, 1)
    try:
        assert plain["row_schedule_handle"] is None and plain["row_schedule_bytes"] == 0
    finally:
        plain.close()


def test_schedule_applies_only_to_the_csr_master_accumulate_operator(monkeypatch):
    calls = []
    master = lambda *args, **kwargs: calls.append(("master", args, kwargs))
    scheduled = lambda *args, **kwargs: calls.append(("scheduled", args, kwargs)) or (args[0], args[6])
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(
        dpointnet_csr_spike_grad_accumulate_csr_master=master,
        dpointnet_csr_spike_grad_accumulate_csr_master_scheduled=scheduled,
    ))
    spikes = tf.zeros((2, 3), tf.float16)
    args = (spikes, None, "metadata", "weights", "basis", "scale", "accumulator")
    ops.spike_grad_accumulate_batched(master, *args, n_post=1, row_schedule="schedule")
    assert calls[-1][0] == "scheduled" and calls[-1][1][7] == "schedule"
    with pytest.raises(ValueError, match="CSR-master accumulate"):
        ops.spike_grad_accumulate_batched(lambda *a, **k: None, *args, n_post=1, row_schedule="schedule")
    assert ops._row_schedule_for({"row_schedule_handle": None}, True) is None
    assert ops._row_schedule_for({"row_schedule_handle": "handle"}, False) is None
    assert ops._row_schedule_for({"row_schedule_handle": "handle"}, True) == "handle"


def _native():
    if not tf.config.list_physical_devices("GPU") or ops._OPS is None:
        pytest.skip("GPU and the native CSR library are required.")
    if not ops.scheduled_accumulate_backward_available():
        pytest.skip("Rebuilt CSR library with the row-scheduled accumulate backward required.")


@pytest.mark.parametrize("batch", [64, 32, 16, 5])
@pytest.mark.parametrize("order", ["reverse", "random", "locality"])
def test_gpu_scheduled_backward_is_bitwise_unscheduled(batch, order):
    """Only row-to-warp assignment changes: spike cotangents (silent rows included)
    and the FP32 weight accumulator are bitwise those of the unscheduled operator."""
    _native()
    from bmtk.simulator.dpointnet.csr_master import CsrMasterLayout

    rng = np.random.default_rng(batch)
    indices, types, position, _, n_sources = _network(n=97, slabs=3, seed=batch)
    n = position.shape[0]
    schedule = {
        "reverse": np.arange(n_sources)[::-1],
        "random": rng.permutation(n_sources),
        "locality": locality_row_schedule(indices, types, n, n_sources, positions=position)[0],
    }[order]
    basis = rng.uniform(-0.3, 0.8, (8, 4)).astype(np.float16)
    weights = rng.normal(0, 0.4, indices.shape[0]).astype(np.float32)
    spikes = (rng.random((batch, n_sources)) < 0.2).astype(np.float16)
    dcurrents = rng.normal(0, 0.5, (batch * n, 4)).astype(np.float16)
    seed = rng.normal(0, 0.1, indices.shape[0]).astype(np.float32)
    built = [
        ops.build_csr_connectivity(
            indices, types, n_sources, n, 8, build_compact_pairs=True, sort_by_target=True,
            host_lifecycle=True, uint8_types=True,
        )
        for _ in range(2)
    ]
    built[1].attach_row_schedule(schedule)
    try:
        results = []
        for conn in built:
            live = CsrMasterLayout(conn, host_lifecycle=True).host_to_live(weights)
            results.append(ops.spike_grad_accumulate_batched(
                ops._OPS.dpointnet_csr_spike_grad_accumulate_csr_master,
                tf.constant(spikes), tf.constant(dcurrents), conn["metadata_handle"],
                tf.constant(live), tf.constant(basis), tf.constant(0.5, tf.float16),
                tf.constant(CsrMasterLayout(conn, host_lifecycle=True).host_to_live(seed)),
                row_schedule=ops._row_schedule_for(conn, True),
                Tindex=tf.uint32, n_post=n, n_edges=conn["n_edges"], n_pairs=conn["n_pairs"],
                use_javier_batch32_backward=True, **ops._uint8_type_options(conn),
            ))
        (plain_dz, plain_dw), (scheduled_dz, scheduled_dw) = results
        assert np.array_equal(plain_dz.numpy().view(np.uint16), scheduled_dz.numpy().view(np.uint16))
        assert np.array_equal(plain_dw.numpy().view(np.uint32), scheduled_dw.numpy().view(np.uint32))
        assert np.any(plain_dz.numpy()[spikes == 0] != 0), "silent sources must keep credit"
    finally:
        for conn in built:
            conn.close()


def test_gpu_rejects_a_schedule_of_the_wrong_length():
    _native()
    conn = ops.build_csr_connectivity(
        [[0, 0], [1, 1]], [0, 1], 2, 2, 2, build_compact_pairs=True, sort_by_target=True,
        host_lifecycle=True, uint8_types=True,
    )
    try:
        wrong = tf.Variable(np.array([0, 1, 2], np.uint32), trainable=False)
        with pytest.raises(tf.errors.InvalidArgumentError, match="row_schedule"):
            ops._OPS.dpointnet_csr_spike_grad_accumulate_csr_master_scheduled(
                tf.ones((2, 2), tf.float16), tf.ones((4, 4), tf.float16), conn["metadata_handle"],
                tf.ones((2,), tf.float32), tf.ones((2, 4), tf.float16), tf.constant(1.0, tf.float16),
                tf.zeros((2,), tf.float32), wrong.handle, Tindex=tf.uint32, n_post=2, n_edges=2,
                n_pairs=conn["n_pairs"], use_javier_batch32_backward=True, **ops._uint8_type_options(conn),
            )
    finally:
        conn.close()


@pytest.mark.parametrize("batch", [64, 32])
def test_gpu_cell_schedule_is_reported_and_leaves_the_direct_route_unchanged(batch, monkeypatch):
    """An explicitly scheduled cell names its schedule, and the direct route's counts,
    traces, final state and gradients equal those of the natural row order."""
    _native()
    from bmtk.simulator.dpointnet.direct_recorded_adjoint import DirectRecordedAdjointRunner
    from test_batched_native_backward import LARGE_BATCH_OPTIONS, _compare, _evaluate, _traces
    from test_direct_recorded_adjoint import _initial_state, _model
    from test_precision_credit import make_cell

    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    cell = make_cell("nest", selective=True, batch_size=batch, hard_reset_gradient_mode="soft_surrogate",
                     use_javier_recurrent_vjp=True, use_locality_row_schedule=True, **LARGE_BATCH_OPTIONS)
    try:
        assert cell._use_locality_row_schedule is True
        assert cell.recurrent_row_schedule.startswith("locality")
        assert cell.recurrent_fused_connectivity["row_schedule_handle"] is not None
        assert cell.acceleration_report["selected"]["use_locality_row_schedule"] is True
        initial = _initial_state(cell, batch)
        model = _model(cell, initial)
        options = dict(sequence_length=7, chunk_size=3, differentiate_inputs=False, count_windows=((0, 7), (1, 6)))
        traces = _traces(cell, 7)
        scheduled, n_exact = _evaluate([DirectRecordedAdjointRunner(model, cell, **options)], cell, initial, model, 7, traces)
        monkeypatch.setattr(ops, "_row_schedule_for", lambda *args: None)
        natural, _ = _evaluate([DirectRecordedAdjointRunner(model, cell, **options)], cell, initial, model, 7, traces)
        _compare(natural[0], scheduled[0], n_exact, f"row schedule batch {batch}")
    finally:
        cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(previous)
