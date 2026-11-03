"""Spike exchange hooks of the direct recorded adjoint.

A network whose recurrent edges from some neurons are moved into a named
spike input, fed back by a loopback exchange (the cell's own spikes), must
reproduce the unsplit network: the same spikes, states, penalties and, per
canonical edge, the same master gradients. This is the one-process form of a
neuron-partitioned network whose exchange is a real transport.
"""

import copy

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available
from bmtk.simulator.dpointnet.direct_recorded_adjoint import (
    DirectRecordedAdjointRunner, SpikeExchange, direct_records_supported,
)
from test_direct_recorded_adjoint import DIRECT_OPTIONS
from test_precision_credit import make_cell

NEURONS, DRIVES, MAX_DELAY, LENGTH, CHUNK = 24, 8, 4, 13, 4
EXCHANGED = "exchanged"


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


class Loopback(SpikeExchange):
    """Returns the cell's own previous-step spikes and their credit unchanged."""

    def __init__(self, log=None):
        self.log = log

    def _record(self, kind, time):
        if self.log is None:
            return
        tf.numpy_function(lambda value: self.log.append((kind, int(value))) or np.int32(0),
                          [time], tf.int32, stateful=True)

    def begin(self):
        if self.log is None:
            return None
        return tf.numpy_function(lambda: self.log.append(("begin", -1)) or np.int32(0), [], tf.int32, stateful=True)

    def forward(self, time, spikes):
        self._record("forward", time)
        return spikes

    def reverse(self, time, credit):
        self._record("reverse", time)
        return credit


def _spec(batch, scale, one_step=False):
    network, _, arguments = make_cell(
        "nest", selective=False, batch_size=batch, hard_reset_gradient_mode="soft_surrogate",
        return_spec=True, **DIRECT_OPTIONS,
    )
    rng = np.random.default_rng(5)
    for key, value in network["node_params"].items():
        network["node_params"][key] = np.concatenate([value] * (NEURONS // 2))
    network["n_nodes"] = NEURONS
    network["node_type_ids"] = np.tile(network["node_type_ids"], NEURONS // 2)
    edges = 120
    targets = rng.integers(0, NEURONS, edges)
    sources = rng.integers(0, NEURONS, edges)
    network["synapses"].update(
        indices=np.stack([targets, sources], axis=1).astype(np.int64),
        weights=rng.uniform(-250.0, 400.0, edges),
        delays=rng.integers(1, MAX_DELAY + 1, edges).astype(np.float64),
        syn_ids=np.zeros(edges, np.int64),
        dense_shape=(NEURONS, NEURONS),
    )
    drive_edges = 40
    drive = {
        "n_inputs": DRIVES,
        "indices": np.stack([rng.integers(0, NEURONS, drive_edges), rng.integers(0, DRIVES, drive_edges)], axis=1),
        "weights": rng.uniform(800.0, 2000.0, drive_edges),
        "delays": (np.ones(drive_edges) if one_step else rng.integers(1, 3, drive_edges).astype(np.float64)),
        "syn_ids": np.zeros(drive_edges, np.int64),
        "input_type": "spikes",
        "options": {"trainable": True},
    }
    arguments.update(max_delay=float(MAX_DELAY), recurrent_dampening_factor=scale, train_recurrent=True)
    return network, drive, arguments


def _cells(batch, scale, moved_sources, one_step=False):
    """Unsplit reference and loopback candidate; ``one_step`` gives moved edges and drive delay 1."""
    network, drive, arguments = _spec(batch, scale, one_step)
    indices = network["synapses"]["indices"]
    moved = np.isin(indices[:, 1], moved_sources)
    if one_step:
        network["synapses"]["delays"][moved] = 1.0
    reference = GLIF3Cell(copy.deepcopy(network), {"drive": copy.deepcopy(drive)}, **arguments)
    local = copy.deepcopy(network)
    for key in ("indices", "weights", "delays", "syn_ids"):
        local["synapses"][key] = network["synapses"][key][~moved]
    exchanged = {
        "n_inputs": NEURONS,
        "indices": indices[moved],
        "weights": network["synapses"]["weights"][moved],
        "delays": network["synapses"]["delays"][moved],
        "syn_ids": network["synapses"]["syn_ids"][moved],
        "input_type": "spikes",
        "options": {"trainable": True},
    }
    candidate = GLIF3Cell(local, {"drive": copy.deepcopy(drive), EXCHANGED: exchanged}, **arguments)
    return reference, candidate, moved


def _initial(cell, batch, history_slots):
    """Shared nonzero initial state; the exchanged input history holds x(-k) = z(-k-1)."""
    state = list(cell.zero_state(batch, cell.compute_dtype))
    state[0] = tf.cast(history_slots, state[0].dtype)
    rng = np.random.default_rng(11)
    state[1] = tf.cast(rng.uniform(0.2, 1.0, state[1].shape), state[1].dtype)
    net = cell.inputs.get(EXCHANGED)
    if net is not None and int(net["history_stop"]) > int(net["history_start"]):
        start, stop = int(net["history_start"]), int(net["history_stop"])
        history = state[7].numpy()
        slots = (stop - start) // NEURONS
        for k in range(slots):
            history[:, start + k * NEURONS:start + (k + 1) * NEURONS] = \
                history_slots[:, (k + 1) * NEURONS:(k + 2) * NEURONS]
        state[7] = tf.constant(history)
    return tuple(state)


def _rollout(cell, batch, history_slots, drive, exchanges=None, final_term=None):
    initial = _initial(cell, batch, history_slots)
    # The exchanged input takes no replay columns; "drive" is declared first in both cells.
    inputs = drive
    runner = DirectRecordedAdjointRunner(
        None, cell, LENGTH, CHUNK, count_windows=((0, LENGTH), (2, 9)), spike_exchanges=exchanges,
    )
    floating = tuple(value for value in initial[:7] if value.dtype.is_floating)
    variables = tuple(cell.trainable_variables)

    @tf.function
    def evaluate():
        with tf.GradientTape() as tape:
            tape.watch(floating)
            output = runner(inputs, initial)
            values = output.counts + output.sequences + tuple(
                value for value in output.final_state[:7] if value.dtype.is_floating
            )
            loss = tf.add_n([
                tf.reduce_sum(tf.square(tf.cast(value, tf.float32)) * (index + 1) * 1e-3)
                for index, value in enumerate(values)
            ])
            if final_term is not None:
                loss += final_term(output.final_state)
        return values, tape.gradient(loss, (*floating, *variables),
                                     unconnected_gradients=tf.UnconnectedGradients.ZERO)

    values, gradients = evaluate()
    n_floating = len(floating)
    return values, gradients[:n_floating], dict(zip([v.name for v in variables], gradients[n_floating:]))


def _canonical(cell, gradients, name):
    if name == "<recurrent>":
        variable, layout = cell.recurrent_weight_values, cell.recurrent_master_layout
    else:
        variable, layout = cell.inputs[name]["input_weight_values"], cell.inputs[name]["master_layout"]
    return layout.host_to_canonical(gradients[variable.name].numpy())


def _history_terms(candidate, batch):
    """Equal losses on the exchanged frames x(L-k) of the candidate's final input history and
    on the reference's final recurrent history slots k, which hold the same spikes z(L-1-k)."""
    net = candidate.inputs[EXCHANGED]
    start, stop = int(net["history_start"]), int(net["history_stop"])
    weights = tf.constant(np.random.default_rng(13).standard_normal((batch, stop - start)), tf.float32)

    def candidate_term(final_state):
        return tf.reduce_sum(weights * tf.cast(final_state[7][:, start:stop], tf.float32))

    def reference_term(final_state):
        return tf.reduce_sum(weights * tf.cast(final_state[0][:, NEURONS:NEURONS + stop - start], tf.float32))

    return reference_term, candidate_term


@pytest.mark.parametrize("history_loss", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("batch", [2, 3])
def test_loopback_exchange_reproduces_unsplit_network(batch, scale, history_loss):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    reference, candidate, moved = _cells(batch, scale, np.arange(NEURONS // 2, NEURONS))
    try:
        assert direct_records_supported(reference) == () and direct_records_supported(candidate) == ()
        assert moved.any() and (~moved).any()
        assert int(candidate.inputs[EXCHANGED]["history_stop"]) > int(candidate.inputs[EXCHANGED]["history_start"])
        rng = np.random.default_rng(3)
        history = (rng.random((batch, NEURONS * MAX_DELAY)) < 0.3).astype(np.float32)
        drive = tf.constant((rng.random((batch, LENGTH, DRIVES)) < 0.4).astype(np.float32))
        reference_term, candidate_term = _history_terms(candidate, batch) if history_loss else (None, None)
        expected, expected_initial, expected_masters = _rollout(
            reference, batch, history, drive, final_term=reference_term,
        )
        log = []
        observed, observed_initial, observed_masters = _rollout(
            candidate, batch, history, drive, {EXCHANGED: Loopback(log)}, final_term=candidate_term,
        )
        counts = expected[0].numpy()
        assert counts.sum() > 0 and (counts < LENGTH).any()
        for target, value in zip(expected[:2], observed[:2]):
            np.testing.assert_array_equal(value.numpy(), target.numpy())
        for target, value in zip(expected[2:], observed[2:]):
            np.testing.assert_allclose(value.numpy(), target.numpy(), rtol=1e-5, atol=1e-6)
        # Initial voltage/ASC/PSC credit and the credit of z(-1) (newest history slot).
        for target, value in zip(expected_initial[1:], observed_initial[1:]):
            np.testing.assert_allclose(value.numpy(), target.numpy(), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(observed_initial[0].numpy()[:, :NEURONS], expected_initial[0].numpy()[:, :NEURONS],
                                   rtol=1e-4, atol=1e-6)
        whole = _canonical(reference, expected_masters, "<recurrent>")
        assert np.any(whole[moved] != 0)
        np.testing.assert_allclose(_canonical(candidate, observed_masters, "<recurrent>"), whole[~moved],
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(_canonical(candidate, observed_masters, EXCHANGED), whole[moved],
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(_canonical(candidate, observed_masters, "drive"),
                                   _canonical(reference, expected_masters, "drive"), rtol=1e-4, atol=1e-6)
        assert log == ([("begin", -1)] + [("forward", t) for t in range(LENGTH)]
                       + [("begin", -1)] + [("reverse", t) for t in reversed(range(LENGTH))])
    finally:
        reference.close_fused_cuda()
        candidate.close_fused_cuda()


def test_spike_exchange_validation():
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    _, candidate, _ = _cells(2, 1.0, np.arange(NEURONS // 2, NEURONS))
    try:
        with pytest.raises(ValueError, match="not a named input"):
            DirectRecordedAdjointRunner(None, candidate, LENGTH, CHUNK, spike_exchanges={"missing": Loopback()})
        with pytest.raises(TypeError, match="no callable"):
            DirectRecordedAdjointRunner(None, candidate, LENGTH, CHUNK, spike_exchanges={EXCHANGED: object()})
        runner = DirectRecordedAdjointRunner(None, candidate, LENGTH, CHUNK, spike_exchanges={EXCHANGED: Loopback()})
        assert runner.replay_columns == {"drive": (0, DRIVES)} and runner.replay_width == DRIVES
        full = tf.zeros((2, LENGTH, int(candidate.inputs_idx[-1])))
        with pytest.raises(ValueError, match="replay columns"):
            runner(full, candidate.zero_state(2, candidate.compute_dtype))
    finally:
        candidate.close_fused_cuda()


def test_loopback_exchange_without_input_history():
    """One-step exchanged edges and drive: the cell has no input-history state at all."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    batch = 2
    reference, candidate, moved = _cells(batch, 1.0, np.arange(NEURONS // 2, NEURONS), one_step=True)
    try:
        assert len(candidate.zero_state(batch, candidate.compute_dtype)) == 7
        rng = np.random.default_rng(3)
        history = (rng.random((batch, NEURONS * MAX_DELAY)) < 0.3).astype(np.float32)
        drive = tf.constant((rng.random((batch, LENGTH, DRIVES)) < 0.4).astype(np.float32))
        expected, _, expected_masters = _rollout(reference, batch, history, drive)
        observed, _, observed_masters = _rollout(candidate, batch, history, drive, {EXCHANGED: Loopback()})
        np.testing.assert_array_equal(observed[0].numpy(), expected[0].numpy())
        assert expected[0].numpy().sum() > 0
        whole = _canonical(reference, expected_masters, "<recurrent>")
        np.testing.assert_allclose(_canonical(candidate, observed_masters, "<recurrent>"), whole[~moved],
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(_canonical(candidate, observed_masters, EXCHANGED), whole[moved],
                                   rtol=1e-4, atol=1e-6)
    finally:
        reference.close_fused_cuda()
        candidate.close_fused_cuda()
