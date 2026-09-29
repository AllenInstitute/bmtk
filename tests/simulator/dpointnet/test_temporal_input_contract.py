"""Public LGN/internal-Poisson input contracts for FP32 temporal adjoints."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from test_precision_credit import make_cell
from bmtk.simulator.dpointnet.rnn_model import RNN
from bmtk.simulator.dpointnet.input_modules.lgn_generator import LGNGenerator
from bmtk.simulator.dpointnet.input_modules.noisy_current import PoissonSpikesInternal
from bmtk.simulator.dpointnet.state_modules.input_state import InitStateFromInputModule
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import fused_glif_state_available
from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def model(mode, fused, continuous=False):
    network, inputs, options = make_cell(
        mode, return_spec=True, temporal_gradient_precision="float32",
        temporal_checkpoint_chunk_size=3, fused=fused, use_fused_cuda=fused,
        use_packed_sm120_backward=False, use_packed_sm120_external_backward=False,
    )
    options.pop("train_recurrent_per_type")
    inputs["drive"]["input_type"] = "current" if continuous else "spikes"
    inputs["background"] = copy.deepcopy(inputs["drive"])
    inputs["background"]["input_type"] = "current"
    rnn = RNN(seq_len=9, batch_size=2, dtype="float16", cell_params=options)
    rnn._recurrent_networks["test"] = SimpleNamespace(to_dict=lambda: copy.deepcopy(network))
    rnn._input_networks["drive"] = SimpleNamespace(
        name="drive", n_spiking_nodes=2, to_dict=lambda: copy.deepcopy(inputs["drive"])
    )
    background = SimpleNamespace(
        name="background", n_spiking_nodes=0, options=inputs["background"]["options"],
        to_dict=lambda: copy.deepcopy(inputs["background"]),
    )
    # Use the real module's nominal-current/options-Poisson configuration.
    poisson = PoissonSpikesInternal(rnn, "poisson", background, firing_rate=2500.0)
    rnn._input_networks["background"] = background
    lgn = object.__new__(LGNGenerator)
    lgn.population_name = "drive"
    rnn._input_generators_mods.update(lgn=lgn, poisson=poisson)
    engine = rnn.set_training(
        rnn=rnn, n_epochs=1, steps_per_epoch=1, training_approach="parallel",
        gradient_checkpointing=True, gradient_checkpoint_chunk_size=3,
    )
    parameter = engine.add_parameters("test", batch_size=2, seq_len=9)
    parameter.add_loss_function(
        "voltage", lambda spikes, voltages, **kwargs: tf.reduce_sum(voltages)
        + tf.reduce_sum(tf.cast(spikes, tf.float32))
    )
    engine._inputs_sig_factory = SimpleNamespace(build=lambda _: [None])
    engine.set_optimizer(tf.keras.optimizers.SGD(learning_rate=0.01))
    rnn.build(training=True)
    return rnn


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("fused", [False, True])
def test_boolean_lgn_and_internal_poisson_public_routes(mode, fused):
    if fused and not fused_glif_state_available():
        pytest.skip("CUDA state operators unavailable")
    rnn = model(mode, fused)
    try:
        assert not rnn.cell._temporal_continuous_inputs
        assert tf.as_dtype(rnn.model.inputs[0].dtype) == tf.bool
        assert rnn.cell.inputs["background"]["input_type"] == "poisson_spikes_internal"
        x = tf.constant(np.tile([[[False, True], [True, False], [True, True]]], (2, 3, 1)))
        rnn.parse_input_mods_from_config = lambda _: []
        module = InitStateFromInputModule(rnn, inputs=[])
        module.spikes_itrs = lambda batch_size: SimpleNamespace(next_spikes=lambda: (x, []))
        initial = module.get_state(max_retries=1)
        assert initial[1].dtype == tf.float32 and initial[4].dtype == tf.float16
        output = tf.function(rnn.run_extractor)(x, initial)
        assert output[0][0].shape == (2, 9, 2)
        # Counts above one distinguish Poisson sampling from Bernoulli events.
        counts = rnn.cell.sample_noise_spikes(64, tf.zeros((64,), tf.int32),
                                             rnn.cell.inputs["background"])
        assert np.max(counts.numpy()) > 1
        engine = rnn.training_engine
        engine.prepare_gradient_checkpointing()
        assert engine._extractor_forward is None
        losses = engine._distributed_train_step([x], [[]], initial)
        assert np.isfinite(losses["__total_loss"].numpy())
        assert int(engine.optimizer.iterations.numpy()) == 1
        rnn.cell.refresh_recurrent_weight_shadow()
        assert all(np.all(np.isfinite(v.numpy())) for v in rnn.model.trainable_variables)
    finally:
        rnn.cleanup()


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_real_continuous_surface_stays_float32_despite_lgn_generator(mode):
    rnn = model(mode, False, continuous=True)
    try:
        assert rnn.cell._temporal_continuous_inputs
        assert tf.as_dtype(rnn.model.inputs[0].dtype) == tf.float32
        x = tf.ones((2, 9, 2), tf.float32) * 0.7
        initial = rnn.zero_state
        output = rnn.run_extractor(x, initial)
        assert np.all(np.isfinite(output[0][1]))
        for dtype in (tf.float16, tf.bool):
            bad = tf.cast(x, dtype)
            with pytest.raises(ValueError, match="float32 model inputs"):
                rnn.model_inputs(bad, initial)
            with pytest.raises(ValueError, match="float32 inputs"):
                TemporalAdjointRunner(rnn.cell)(bad, initial)
    finally:
        rnn.cleanup()
