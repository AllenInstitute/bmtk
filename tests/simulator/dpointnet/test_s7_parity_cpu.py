"""CPU contracts for the S7 Javier-parity sparse paths."""

import copy

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.custom_ops import build_csr_connectivity
from test_precision_credit import make_cell
from test_csr_fp32_batch32 import _fixture, _oracle, _packed_cpu, _assert_vjp


def _make_uniform_delay_cell(*, batch_size, uniform_projection):
    network, inputs, options = make_cell(
        "nest",
        selective=False,
        return_spec=True,
        train_recurrent=False,
        use_fused_cuda=False,
        track_voltage_penalty=False,
    )
    inputs = copy.deepcopy(inputs)
    inputs["drive"]["input_type"] = "spikes"
    inputs["drive"]["delays"] = np.array([2.0, 2.0], np.float32)
    inputs["drive"]["options"] = {"trainable": False}
    options.update(
        batch_size=batch_size,
        use_uniform_input_delay_projection=uniform_projection,
        track_voltage_penalty=False,
        return_voltage_sequences=True,
    )
    return GLIF3Cell(copy.deepcopy(network), inputs, **options), inputs["drive"]


def _input_current_trace(cell, frames):
    state = cell.zero_state(frames.shape[1], tf.float32)
    currents = []
    for frame in frames:
        current, history = cell._project_step_currents(tf.constant(frame), state)
        currents.append(current.numpy())
        if cell._input_history_size:
            state = state[:7] + (tf.concat(history, axis=1),)
    return np.asarray(currents)


def _dense_lgn_oracle(cell, input_spec, frames):
    weights = cell.inputs["drive"]["input_weight_values"].numpy().astype(np.float64)
    basis = cell.synaptic_basis_weights.numpy().astype(np.float64)
    indices = np.asarray(input_spec["indices"], dtype=np.int64)
    syn_ids = np.asarray(input_spec["syn_ids"], dtype=np.int64)
    out = np.zeros((frames.shape[0], frames.shape[1], cell._n_neurons * 4), np.float64)
    for step in range(1, frames.shape[0]):
        delayed = frames[step - 1].astype(np.float64)
        for edge, (post, pre) in enumerate(indices):
            out[step, :, post * 4 : post * 4 + 4] += (
                delayed[:, pre, None] * weights[edge] * basis[syn_ids[edge]]
            )
    return out.astype(np.float32)


@pytest.mark.parametrize("batch_size", [3, 32])
def test_uniform_lgn_delay_projection_matches_expanded_dense_fp64(batch_size):
    expanded, input_spec = _make_uniform_delay_cell(
        batch_size=batch_size, uniform_projection=False
    )
    uniform, _ = _make_uniform_delay_cell(
        batch_size=batch_size, uniform_projection=True
    )
    frames = np.zeros((4, batch_size, input_spec["n_inputs"]), np.float32)
    frames[1, :, :] = 1.0
    frames[2, :, 0] = (np.arange(batch_size) % 2).astype(np.float32)
    frames[3, :, 1] = 1.0

    expanded_trace = _input_current_trace(expanded, frames)
    uniform_trace = _input_current_trace(uniform, frames)
    oracle = _dense_lgn_oracle(uniform, input_spec, frames)

    np.testing.assert_allclose(uniform_trace, expanded_trace, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(uniform_trace, oracle, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(uniform_trace[0], 0.0)
    np.testing.assert_array_equal(uniform_trace[1], 0.0)
    assert np.any(uniform_trace[2] != 0.0)


def test_forward_run_aggregation_marks_only_repeated_same_target_rows():
    repeated = np.array([[0, 0], [0, 0], [1, 0], [1, 1]], np.int64)
    unique = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.int64)
    repeated_conn = build_csr_connectivity(
        repeated, np.array([0, 1, 0, 0]), 2, 2, 2, sort_by_target=True
    )
    unique_conn = build_csr_connectivity(
        unique, np.array([0, 1, 0, 1]), 2, 2, 2, sort_by_target=True
    )
    try:
        assert repeated_conn["has_repeated_targets"] is True
        assert unique_conn["has_repeated_targets"] is False
    finally:
        repeated_conn.close()
        unique_conn.close()


def test_javier_vjp_cpu_retains_silent_rows_and_nonzero_accumulator():
    fixture = _fixture("binary")
    scale = -0.375
    _, expected_ds, expected_dw, ds_bound, dw_bound = _oracle(
        fixture, direct=True, scale=scale
    )
    ds, dw = _packed_cpu(fixture, direct=True, scale=scale)
    accumulator = np.linspace(-0.25, 0.25, dw.size, dtype=np.float32)
    accumulated = accumulator + dw

    _assert_vjp(ds, expected_ds, ds_bound)
    _assert_vjp(dw, expected_dw, dw_bound)
    assert np.any(ds[:, 4] != 0.0), "silent source rows keep spike adjoints"
    _assert_vjp(accumulated - accumulator, expected_dw, dw_bound)
    assert np.any(accumulated != 0.0)
