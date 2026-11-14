"""Batches above 32 on the native direct route: sliced backward launches.

The native backward kernels take at most 32 samples per launch; the wrappers
slice larger batches and concatenate/accumulate. Slicing must not change
results, and batch 64 must resolve to the direct recorded adjoint.
"""

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops, fused_cuda_available
from bmtk.simulator.dpointnet.direct_recorded_adjoint import DirectRecordedAdjointRunner, direct_records_supported
from bmtk.simulator.dpointnet.spike_statistics import SampledSpikeTraces, SpikeStatisticsRunner
from test_direct_recorded_adjoint import DIRECT_OPTIONS, _initial_state, _model
from test_precision_credit import make_cell


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def _gpu_required():
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")


# Explicit small-batch-only forward routes (device active queue) are rejected
# above 32 samples by design; the remaining explicit options stay.
LARGE_BATCH_OPTIONS = {k: v for k, v in DIRECT_OPTIONS.items() if k != "use_device_active_queue_forward"}


def test_batch_slices():
    assert csr_spike_ops._batch_slices(32) == [(0, 32)]
    assert csr_spike_ops._batch_slices(64) == [(0, 32), (32, 64)]
    assert csr_spike_ops._batch_slices(70) == [(0, 32), (32, 64), (64, 70)]


def _evaluate(runners, cell, initial, model, length, traces):
    inputs = tf.zeros((int(initial[0].shape[0]), length, 0), cell.compute_dtype)
    floating = tuple(value for value in initial if value.dtype.is_floating)

    @tf.function
    def evaluate():
        results = []
        for runner in runners:
            with tf.GradientTape() as tape:
                tape.watch(floating)
                output = runner(inputs, initial, sampled_traces=traces)
                values = output.counts + output.traces + output.sequences + tuple(
                    value for value in output.final_state if value.dtype.is_floating
                )
                loss = tf.add_n([
                    tf.reduce_sum(tf.square(tf.cast(value, tf.float32))) * (index + 1)
                    for index, value in enumerate(values)
                ])
            gradients = tape.gradient(
                loss, (*floating, *model.trainable_variables), unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            results.append((values, gradients))
        return results

    results = evaluate()
    return results, len(results[0][0]) + len(floating)


def _traces(cell, length):
    return (
        SampledSpikeTraces(tf.constant([[0, 0], [1, 0]], tf.int64), tf.ones((2, 1, 2), cell.compute_dtype),
                           (1, length), (0, 2)),
        SampledSpikeTraces(tf.constant([[1, 1]], tf.int64), tf.constant([[[1, 0]]], cell.compute_dtype),
                           (0, 4), (2, 3)),
    )


def _compare(reference, observed, n_exact, label):
    for index, (expected, value) in enumerate(zip(tf.nest.flatten(reference), tf.nest.flatten(observed))):
        if index < n_exact:
            np.testing.assert_array_equal(value.numpy(), expected.numpy(), err_msg=f"{label} tensor {index}")
        else:
            np.testing.assert_allclose(value, expected, rtol=1e-6, atol=0, err_msg=f"{label} master {index}")


@pytest.mark.parametrize("selective", [False, True])
def test_sliced_backward_matches_single_launch(selective, monkeypatch):
    """Forcing 4-sample slices at batch 8 reproduces the single 8-sample launch."""
    _gpu_required()
    cell = make_cell("nest", selective=selective, batch_size=8, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        initial = _initial_state(cell, 8)
        model = _model(cell, initial)
        options = dict(sequence_length=7, chunk_size=3, differentiate_inputs=False, count_windows=((0, 7), (1, 6)))
        single = DirectRecordedAdjointRunner(model, cell, **options)
        sliced = DirectRecordedAdjointRunner(model, cell, **options)
        traces = _traces(cell, 7)
        results, n_exact = _evaluate([single], cell, initial, model, 7, traces)
        monkeypatch.setattr(csr_spike_ops, "NATIVE_BACKWARD_BATCH", 4)
        assert csr_spike_ops._batch_slices(8) == [(0, 4), (4, 8)]
        observed, _ = _evaluate([sliced], cell, initial, model, 7, traces)
        _compare(results[0], observed[0], n_exact, "sliced")
        assert all(np.any(value.numpy() != 0) for value in observed[0][1][-3:])
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("offload", [(), ("voltage", "history")])
def test_batch_64_resolves_direct_route_and_matches_reference(offload):
    """Batch 64 is admitted by the automatic policy (joint sweep, CSR masters,
    fused accumulation) and the direct route matches the checkpointed reference."""
    _gpu_required()
    cell = make_cell("nest", selective=True, batch_size=64, hard_reset_gradient_mode="soft_surrogate", **LARGE_BATCH_OPTIONS)
    try:
        assert direct_records_supported(cell) == ()
        assert cell._use_combined_current_sweep and cell._csr_master_weights and cell.use_fused_recurrent_accumulation
        initial = _initial_state(cell, 64)
        model = _model(cell, initial)
        options = dict(sequence_length=7, chunk_size=3, differentiate_inputs=False, count_windows=((0, 7), (1, 6)))
        reference = SpikeStatisticsRunner(model, n_sequence_outputs=2, pack_spike_checkpoints=True, **options)
        direct = DirectRecordedAdjointRunner(model, cell, record_offload=offload, **options)
        results, n_exact = _evaluate([reference, direct], cell, initial, model, 7, _traces(cell, 7))
        _compare(results[0], results[1], n_exact, f"batch64 {offload}")
        assert all(np.any(value.numpy() != 0) for value in results[1][1][-3:])
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("batch", [64, 100])
def test_joint_sweep_admission_no_longer_capped_at_32(batch):
    """The automatic joint-sweep policy admits any positive batch; the per-launch
    bound lives in the sliced backward wrappers."""
    from bmtk.simulator.dpointnet import acceleration

    diagnostics = {}
    context = {"standard_cell": True, "single_replica": True, "spike_inputs": True, "recurrent_edges": 1000,
               "n_post": 1000, "source_rows_upper_bound": 2000, "max_edges": 100000}
    acceleration.select_performance_projection(
        {}, context, compute_dtype=tf.float16, variable_dtype=tf.float32, batch_size=batch, basis_width=4,
        architecture=86, diagnostics=diagnostics,
    )
    assert diagnostics[f"positive batch (found {batch})"] is True
    assert not any("batch1..32" in name for name in diagnostics)
    assert acceleration.JOINT_SWEEP_MAX_BATCH == csr_spike_ops.NATIVE_BACKWARD_BATCH == 32
