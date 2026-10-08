import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.cell_models.state_rnn import (
    ExplicitStateRNN, _rollout_context,
)
from bmtk.simulator.dpointnet import temporal_adjoint
from test_precision_credit import make_cell


class _Cell(tf.keras.layers.Layer):
    state_size = 1
    output_size = 1
    temporal_checkpoint_chunk_size = 2
    temporal_pack_spike_checkpoints = False
    _use_prepacked_nest_coefficients = True

    def __init__(self, precision, failure=None):
        super().__init__(dtype="float32", name="lifetime_cell")
        self.temporal_gradient_precision = precision
        self.failure = failure
        self.noise_seed = tf.Variable([11, 17], dtype=tf.int32, trainable=False)
        self.prepared = (tf.constant([3.0]),)
        self.prepare_calls = 0

    def validate_state_precision(self, states):
        pass

    def prepare_rollout_nest_coefficients(self):
        self.prepare_calls += 1
        if self.failure == "coefficients":
            self._rollout_nest_coefficients = self.prepared
            raise RuntimeError("coefficient preparation failed")
        return self.prepared

    def call(self, inputs, states):
        assert self._rollout_nest_coefficients[0] is self.prepared[0]
        if self.failure == "execution":
            raise RuntimeError("rollout execution failed")
        value = inputs + states[0]
        return value, [value]


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def _install_runner(monkeypatch, failure=None):
    class Runner:
        def __init__(self, cell, **kwargs):
            self.cell = cell
            assert cell._rollout_nest_coefficients[0] is cell.prepared[0]
            if failure == "runner":
                raise RuntimeError("runner preparation failed")

        def __call__(self, sequences, states):
            if failure == "execution":
                raise RuntimeError("rollout execution failed")
            return sequences, states

    monkeypatch.setattr(temporal_adjoint, "TemporalAdjointRunner", Runner)


def _invoke(layer, graph=False, mask=None):
    def invoke():
        return layer.call(
            tf.ones((2, 3, 1)), initial_state=[tf.zeros((2, 1))], mask=mask
        )

    return tf.function(invoke)() if graph else invoke()


@pytest.mark.parametrize(
    "precision,failure",
    [(precision, failure) for precision in ("compute", "float32")
     for failure in (None, "coefficients", "execution", "runner")
     if precision == "float32" or failure != "runner"],
)
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_rollout_attributes_restored_at_every_call_boundary(
    precision, graph, existing, failure, monkeypatch,
):
    cell = _Cell(precision, failure)
    previous = {
        "_rollout_nest_coefficients": tf.constant([9.0]),
        "_rollout_noise_seed": tf.constant([29, 31], tf.int32),
    }
    if existing:
        for name, value in previous.items():
            setattr(cell, name, value)
    if precision == "float32":
        _install_runner(monkeypatch, failure)
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    if failure is None:
        output = _invoke(layer, graph)
        assert output[0].shape == (2, 3, 1)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            _invoke(layer, graph)
    for name, value in previous.items():
        if existing:
            assert getattr(cell, name) is value
        else:
            assert not hasattr(cell, name)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("time_major", [False, True])
def test_fp32_invalid_options_rejected_before_preparation(graph, time_major):
    cell = _Cell("float32")
    coefficients = tf.constant([9.0])
    cell._rollout_nest_coefficients = coefficients
    layer = ExplicitStateRNN(cell)
    if time_major:
        layer.time_major = True
    mask = None if time_major else tf.ones((2, 3), tf.bool)
    with pytest.raises(ValueError, match="unmasked batch-major"):
        _invoke(layer, graph, mask)
    assert cell.prepare_calls == 0
    assert cell._rollout_nest_coefficients is coefficients
    assert not hasattr(cell, "_rollout_noise_seed")


def test_noise_snapshot_failure_restores_prepared_coefficients(monkeypatch):
    cell = _Cell("compute")
    previous = tf.constant([9.0])
    cell._rollout_nest_coefficients = previous
    original_identity = tf.identity

    def identity(value, *args, **kwargs):
        if value is cell.noise_seed:
            raise RuntimeError("noise snapshot failed")
        return original_identity(value, *args, **kwargs)

    monkeypatch.setattr(tf, "identity", identity)
    with pytest.raises(RuntimeError, match="noise snapshot failed"):
        _invoke(ExplicitStateRNN(cell))
    assert cell.prepare_calls == 1
    assert cell._rollout_nest_coefficients is previous
    assert not hasattr(cell, "_rollout_noise_seed")


def test_nested_rollout_context_restores_outer_snapshot():
    cell = _Cell("compute")
    with _rollout_context(cell):
        outer_seed = cell._rollout_noise_seed
        np.testing.assert_array_equal(outer_seed, [11, 17])
        cell.noise_seed.assign([19, 23])
        with _rollout_context(cell):
            np.testing.assert_array_equal(cell._rollout_noise_seed, [19, 23])
            assert cell._rollout_noise_seed is not outer_seed
        assert cell._rollout_noise_seed is outer_seed
    assert not hasattr(cell, "_rollout_noise_seed")
    assert not hasattr(cell, "_rollout_nest_coefficients")


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_rejected_rollout_does_not_pin_old_poisson_stream(mode):
    cell = make_cell(
        mode, noise=True, temporal_gradient_precision="float32",
        use_fused_cuda=False,
    )
    layer = ExplicitStateRNN(cell, return_sequences=True)
    seed = cell.noise_seed.numpy().copy()
    with pytest.raises(ValueError, match="unmasked batch-major"):
        layer.call(
            tf.zeros((2, 3, 2), tf.bool),
            initial_state=cell.zero_state(2, tf.float32),
            mask=tf.ones((2, 3), tf.bool),
        )
    assert getattr(cell, "_rollout_noise_seed", None) is None
    cell.advance_noise_seed()
    assert not np.array_equal(cell.noise_seed.numpy(), seed)
    network = cell.inputs["drive"]
    first = cell.sample_noise_spikes(128, tf.zeros((128,), tf.int32), network)
    current_seed = cell.noise_seed.numpy().copy()
    cell.noise_seed.assign(seed)
    old = cell.sample_noise_spikes(128, tf.zeros((128,), tf.int32), network)
    cell.noise_seed.assign(current_seed)
    repeated = cell.sample_noise_spikes(128, tf.zeros((128,), tf.int32), network)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first.numpy(), old.numpy())
