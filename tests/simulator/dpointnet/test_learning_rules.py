from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.learning_rules import (
    EPropLearningRule,
    LearningRuleObservations,
    LearningRules,
    LocalRateHomeostasisLearningRule,
    ModulatedEligibilityLearningRule,
    NeuronLocalThreeFactorLearningRule,
    PairSTDPLearningRule,
    ThreeFactorLearningRule,
    WeightSurface,
    register_learning_rule,
)
from bmtk.simulator.dpointnet.learning_rules.base import LearningRule
from bmtk.simulator.dpointnet.training import TrainingEngine
from bmtk.simulator.dpointnet import optimizers


def _fake_rnn():
    recurrent = tf.Variable([1.0], dtype=tf.float32, trainable=True)
    input_weight = tf.Variable([2.0], dtype=tf.float32, trainable=True)
    cell = SimpleNamespace(
        _n_neurons=2,
        _node_type_ids=np.array([10, 20]),
        max_delay=1,
        _dampening_factor=0.5,
        _voltage_gradient_dampening=1.0,
        _hard_reset=False,
        _n_syn_basis=1,
        _dt=1.0,
        _lr_scale=1.0,
        v_th=tf.constant(1.0),
        synaptic_basis_weights=tf.constant([[1.0]]),
        syn_decay=tf.ones([1, 2]),
        psc_initial=tf.ones([1, 2]),
        decay=tf.zeros([2]),
        current_factor=tf.ones([2]),
        recurrent_weight_values=recurrent,
        recurrent_indices=tf.constant([[0, 1]], dtype=tf.int64),
        syn_ids=tf.constant([0], dtype=tf.int64),
        inputs_idx=np.array([0, 1]),
        inputs={
            "lgn": {
                "input_dim": 1,
                "input_type": "spikes",
                "input_weight_values": input_weight,
                "input_indices": tf.constant([[1, 0]], dtype=tf.int64),
                "input_syn_ids": tf.constant([0], dtype=tf.int64),
            }
        },
    )
    return SimpleNamespace(_cell=cell), recurrent, input_weight


def _observations():
    return LearningRuleObservations(
        input_spikes=tf.constant([[[2.0], [0.0], [0.0]]]),
        spikes=tf.constant([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]),
        voltages=tf.ones([1, 3, 2], dtype=tf.float32),
        initial_state=(tf.constant([[0.0, 1.0]]),),
        spike_learning_signal=tf.constant([[[2.0, 1.0], [3.0, 4.0], [4.0, 5.0]]]),
        voltage_learning_signal=tf.zeros([1, 3, 2], dtype=tf.float32),
        direct_weight_gradients=(tf.zeros([1]), tf.zeros([1])),
    )


def test_eprop_computes_delayed_recurrent_and_input_updates():
    rnn, recurrent, input_weight = _fake_rnn()
    rule = EPropLearningRule(
        edge_chunk_size=1,
        surfaces=("<recurrent>", "lgn"),
    )
    rule.build(rnn)

    updates = {
        surface.name: gradient
        for surface, gradient in rule.compute_updates(_observations())
    }

    np.testing.assert_allclose(updates["<recurrent>"].numpy(), [2.0])
    np.testing.assert_allclose(updates["lgn"].numpy(), [5.0])
    assert rule.weight_surfaces[0].variable is recurrent
    assert rule.weight_surfaces[1].variable is input_weight


def test_eprop_applies_bounds_and_tracks_serializable_step(tmp_path):
    variable = tf.Variable([0.9], dtype=tf.float32, trainable=True)
    surface = WeightSurface(
        name="<recurrent>",
        variable=variable,
        indices=tf.constant([[0, 0]], dtype=tf.int64),
        synapse_types=tf.constant([0], dtype=tf.int64),
        source="recurrent",
    )
    rule = EPropLearningRule(min_weight=-0.5, max_weight=0.5)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    rule.apply_updates(optimizer, ((surface, tf.constant([-1.0])),))
    rule.on_global_step_end()

    np.testing.assert_allclose(variable.numpy(), [0.5])
    assert int(rule.update_step.numpy()) == 1
    checkpoint_path = tf.train.Checkpoint(rule=rule).save(str(tmp_path / "rule"))

    restored = EPropLearningRule()
    tf.train.Checkpoint(rule=restored).restore(checkpoint_path).assert_consumed()
    assert int(restored.update_step.numpy()) == 1


def test_eprop_config_round_trip():
    rule = EPropLearningRule(
        edge_chunk_size=64,
        surfaces=("lgn",),
        learning_signal_clip=2.0,
        gradient_clip_norm=3.0,
    )
    config = rule.get_config()
    name = config.pop("name")
    restored = LearningRules().get_rule(name)(**config)

    assert restored.get_config() == rule.get_config()


def test_surface_validation_rejects_each_irrelevant_request():
    rnn, _, _ = _fake_rnn()
    with pytest.raises(ValueError, match="Unknown learning-rule surface.*missing"):
        EPropLearningRule(surfaces=("<recurrent>", "missing")).build(rnn)

    rnn._cell.inputs["lgn"]["input_weight_values"] = tf.Variable([2.0], trainable=False)
    with pytest.raises(ValueError, match="not trainable: lgn"):
        EPropLearningRule(surfaces=("lgn",)).build(rnn)

    rnn._cell.inputs["lgn"]["input_weight_values"] = tf.Variable([2.0], trainable=True)
    rnn._cell.inputs["lgn"]["input_type"] = "poisson_spikes_internal"
    with pytest.raises(ValueError, match="cannot observe internally generated"):
        EPropLearningRule(surfaces=("lgn",)).build(rnn)


def test_three_factor_selects_spike_or_voltage_modulator():
    rnn, _, _ = _fake_rnn()
    observations = _observations()
    spike_rule = ThreeFactorLearningRule(
        signal="spike", edge_chunk_size=1, surfaces=("<recurrent>",)
    )
    spike_rule.build(rnn)
    voltage_rule = ThreeFactorLearningRule(
        signal="voltage", edge_chunk_size=1, surfaces=("<recurrent>",)
    )
    voltage_rule.build(rnn)

    spike_gradient = spike_rule.compute_updates(observations)[0][1]
    voltage_observations = replace(
        observations,
        voltage_learning_signal=tf.ones_like(observations.voltages),
    )
    voltage_gradient = voltage_rule.compute_updates(voltage_observations)[0][1]

    np.testing.assert_allclose(spike_gradient.numpy(), [2.0])
    np.testing.assert_allclose(voltage_gradient.numpy(), [1.0])
    with pytest.raises(ValueError, match="Unknown three-factor signal"):
        ThreeFactorLearningRule(signal="irrelevant")


def test_three_factor_config_round_trip():
    rule = ThreeFactorLearningRule(
        signal="voltage",
        spike_signal_scale=2.0,
        voltage_signal_scale=3.0,
        surfaces=("<recurrent>",),
    )
    config = rule.get_config()
    name = config.pop("name")
    rebuilt = LearningRules().get_rule(name)(**config)

    assert rebuilt.get_config() == rule.get_config()


def test_pair_stdp_uses_only_local_spikes_and_potentiates_causal_pair():
    rnn, _, _ = _fake_rnn()
    observations = replace(
        _observations(),
        spikes=tf.constant([[[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]]),
        initial_state=(tf.zeros([1, 2]),),
        spike_learning_signal=tf.zeros([1, 3, 2]),
        voltage_learning_signal=tf.zeros([1, 3, 2]),
    )
    rule = PairSTDPLearningRule(surfaces=("<recurrent>",), a_plus=1.0, a_minus=0.0)
    rule.build(rnn)

    gradient = rule.compute_updates(observations)[0][1]

    assert float(gradient.numpy()[0]) < 0.0


def test_local_rate_homeostasis_increases_input_to_silent_postsynaptic_cell():
    rnn, _, _ = _fake_rnn()
    observations = replace(
        _observations(),
        spikes=tf.constant([[[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]),
        initial_state=(tf.zeros([1, 2]),),
        spike_learning_signal=tf.zeros([1, 3, 2]),
        voltage_learning_signal=tf.zeros([1, 3, 2]),
    )
    rule = LocalRateHomeostasisLearningRule(
        target_rate_hz=20.0, surfaces=("<recurrent>",)
    )
    rule.build(rnn)

    gradient = rule.compute_updates(observations)[0][1]

    assert float(gradient.numpy()[0]) < 0.0


@pytest.mark.parametrize(
    "rule",
    [
        PairSTDPLearningRule(surfaces=("<recurrent>",)),
        LocalRateHomeostasisLearningRule(
            target_rate_hz=20.0, surfaces=("<recurrent>",)
        ),
    ],
)
def test_strict_local_updates_trace_in_graph_mode(rule):
    rnn, _, _ = _fake_rnn()
    rnn._cell._dt = tf.constant(1.0)
    rule.build(rnn)

    @tf.function
    def compute():
        return rule.compute_updates(_observations())[0][1]

    gradient = compute()

    assert np.all(np.isfinite(gradient.numpy()))


@pytest.mark.parametrize(
    "rule",
    [
        PairSTDPLearningRule(weight_dependence="multiplicative"),
        LocalRateHomeostasisLearningRule(
            target_rate_hz=5.0, update="multiplicative_post_rate_scaling"
        ),
    ],
)
def test_strict_local_rule_config_round_trip(rule):
    config = rule.get_config()
    name = config.pop("name")
    rebuilt = LearningRules().get_rule(name)(**config)

    assert rebuilt.get_config() == rule.get_config()


def test_single_channel_modulator_broadcasts_population_mean():
    rnn, _, _ = _fake_rnn()
    rule = ModulatedEligibilityLearningRule(
        n_channels=1, signal="voltage", surfaces=("<recurrent>",)
    )
    rule.build(rnn)
    observations = replace(
        _observations(),
        voltage_learning_signal=tf.constant([[[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]]]),
    )

    local_factor = rule._local_factor(observations, tf.ones_like(observations.voltages))

    np.testing.assert_allclose(
        local_factor.numpy(), [[[2.0, 2.0], [3.0, 3.0], [6.0, 6.0]]]
    )


def test_finite_channel_partition_shares_modulator_only_within_channel():
    rnn, _, _ = _fake_rnn()
    rule = ModulatedEligibilityLearningRule(
        n_channels=2,
        signal="voltage",
        channel_projection="fixed_balanced_partition",
        surfaces=("<recurrent>",),
    )
    rule.build(rnn)
    observations = replace(
        _observations(),
        voltage_learning_signal=tf.constant([[[1.0, 3.0], [2.0, 4.0], [5.0, 7.0]]]),
    )

    local_factor = rule._local_factor(observations, tf.ones_like(observations.voltages))

    np.testing.assert_allclose(
        local_factor.numpy(), observations.voltage_learning_signal.numpy()
    )


def test_modulated_eligibility_config_round_trip():
    rule = ModulatedEligibilityLearningRule(
        n_channels=4,
        signal="spike",
        channel_projection="cell_class_partition",
        surfaces=("<recurrent>",),
    )
    config = rule.get_config()
    name = config.pop("name")
    rebuilt = LearningRules().get_rule(name)(**config)

    assert rebuilt.get_config() == rule.get_config()


def test_neuron_local_rate_error_uses_targets_not_global_loss_derivatives():
    rnn, _, _ = _fake_rnn()
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_rate_error",
        pool_a_start=0,
        pool_a_end=1,
        pool_b_start=1,
        pool_b_end=2,
        cue_duration_ms=1.0,
        response_window_ms=2.0,
        high_target_rate_hz=1000.0,
        low_target_rate_hz=0.0,
        tau_rate_ms=1.0,
        surfaces=("<recurrent>",),
    )
    rule.build(rnn)
    observations = replace(
        _observations(),
        spikes=tf.zeros([1, 3, 2], dtype=tf.float32),
        targets={
            "class_label": tf.constant([0], dtype=tf.int32),
            "delay_ms": tf.constant([0.0], dtype=tf.float32),
        },
    )

    local_factor = rule._local_factor(observations, tf.ones_like(observations.spikes))
    changed_global_signals = replace(
        observations,
        spike_learning_signal=tf.fill([1, 3, 2], 1000.0),
        voltage_learning_signal=tf.fill([1, 3, 2], -1000.0),
    )

    assert np.all(local_factor.numpy()[:, 0, :] == 0.0)
    assert np.all(local_factor.numpy()[:, 1:, 0] < 0.0)
    assert np.all(local_factor.numpy()[:, 1:, 1] == 0.0)
    np.testing.assert_allclose(
        rule._local_factor(
            changed_global_signals, tf.ones_like(observations.spikes)
        ).numpy(),
        local_factor.numpy(),
    )

    direct_gradient_changed = replace(
        observations,
        direct_weight_gradients=(tf.constant([1000.0]), tf.constant([1000.0])),
    )
    np.testing.assert_allclose(
        rule.compute_updates(direct_gradient_changed)[0][1].numpy(),
        rule.compute_updates(observations)[0][1].numpy(),
    )


def test_neuron_local_rate_error_supports_population_homeostasis():
    rnn, _, _ = _fake_rnn()
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_rate_error",
        target_rate_hz=1000.0,
        pool_a_start=0,
        pool_a_end=1,
        pool_b_start=1,
        pool_b_end=2,
        tau_rate_ms=1.0,
        surfaces=("<recurrent>",),
    )
    rule.build(rnn)
    observations = replace(
        _observations(),
        spikes=tf.zeros([1, 3, 2], dtype=tf.float32),
    )

    local_factor = rule._local_factor(observations, tf.ones_like(observations.spikes))

    assert np.all(local_factor.numpy() < 0.0)


def test_neuron_local_config_round_trip():
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_voltage_error",
        target_rate_hz=5.0,
        high_target_voltage_offset=0.2,
        low_target_voltage_offset=-0.8,
        surfaces=("<recurrent>",),
    )
    config = rule.get_config()
    name = config.pop("name")
    rebuilt = LearningRules().get_rule(name)(**config)

    assert rebuilt.get_config() == rule.get_config()


def test_neuron_local_voltage_error_accepts_scalar_threshold():
    rnn, _, _ = _fake_rnn()
    rnn._cell.v_th = tf.constant(1.0)
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_voltage_error",
        target_rate_hz=5.0,
        high_target_voltage_offset=0.5,
        pool_a_start=0,
        pool_a_end=1,
        pool_b_start=1,
        pool_b_end=2,
        surfaces=("<recurrent>",),
    )
    rule.build(rnn)

    local_factor = rule._local_factor(
        _observations(), tf.ones_like(_observations().spikes)
    )

    assert local_factor.shape == (1, 3, 2)
    np.testing.assert_allclose(local_factor.numpy(), -0.5)


def test_neuron_local_cue_memory_persists_and_gates_response_error():
    rnn, _, _ = _fake_rnn()
    observations = replace(
        _observations(),
        spikes=tf.constant([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]),
        voltages=tf.ones([1, 4, 2], dtype=tf.float32),
        spike_learning_signal=tf.zeros([1, 4, 2], dtype=tf.float32),
        voltage_learning_signal=tf.zeros([1, 4, 2], dtype=tf.float32),
        targets={
            "class_label": tf.constant([0], dtype=tf.int32),
            "delay_ms": tf.constant([1.0], dtype=tf.float32),
        },
    )
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_rate_error",
        pool_a_start=0,
        pool_a_end=1,
        pool_b_start=1,
        pool_b_end=2,
        cue_duration_ms=1.0,
        response_window_ms=2.0,
        high_target_rate_hz=1000.0,
        low_target_rate_hz=0.0,
        tau_rate_ms=0.01,
        cue_memory_tau_ms=100.0,
        cue_memory_gain=10.0,
        cue_memory_signal="spike",
        surfaces=("<recurrent>",),
    )
    rule.build(rnn)

    memory = rule._cue_memory(observations, tf.ones_like(observations.spikes))
    local_factor = rule._local_factor(observations, tf.ones_like(observations.spikes))

    assert float(memory.numpy()[0, 2, 0]) > 0.0
    assert float(memory.numpy()[0, 2, 1]) == 0.0
    assert float(local_factor.numpy()[0, 2, 0]) < -1.0
    assert float(local_factor.numpy()[0, 1, 0]) == 0.0


def test_neuron_local_synaptic_cue_eligibility_bridges_delay():
    rnn, _, _ = _fake_rnn()
    observations = replace(
        _observations(),
        spikes=tf.constant([[[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        voltages=tf.ones([1, 4, 2], dtype=tf.float32),
        spike_learning_signal=tf.zeros([1, 4, 2], dtype=tf.float32),
        voltage_learning_signal=tf.zeros([1, 4, 2], dtype=tf.float32),
        targets={
            "class_label": tf.constant([0], dtype=tf.int32),
            "delay_ms": tf.constant([1.0], dtype=tf.float32),
        },
    )
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_voltage_error",
        pool_a_start=0,
        pool_a_end=1,
        pool_b_start=1,
        pool_b_end=2,
        cue_duration_ms=1.0,
        response_window_ms=2.0,
        high_target_voltage_offset=0.5,
        low_target_voltage_offset=-1.0,
        cue_memory_mode="synaptic_eligibility",
        cue_memory_tau_ms=100.0,
        cue_memory_gain=1.0,
        cue_memory_floor=0.0,
        surfaces=("<recurrent>",),
        edge_chunk_size=1,
    )
    rule.build(rnn)

    gradient = rule.compute_updates(observations)[0][1]
    no_cue = replace(observations, spikes=tf.zeros_like(observations.spikes))

    assert float(gradient.numpy()[0]) < 0.0
    np.testing.assert_allclose(rule.compute_updates(no_cue)[0][1].numpy(), 0.0)


def test_neuron_local_synaptic_spike_memory_requires_local_coincidence():
    rnn, _, _ = _fake_rnn()
    observations = replace(
        _observations(),
        spikes=tf.constant([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        voltages=tf.ones([1, 4, 2], dtype=tf.float32),
        targets={
            "class_label": tf.constant([0], dtype=tf.int32),
            "delay_ms": tf.constant([1.0], dtype=tf.float32),
        },
    )
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_voltage_error",
        pool_a_start=0,
        pool_a_end=1,
        pool_b_start=1,
        pool_b_end=2,
        cue_duration_ms=1.0,
        response_window_ms=2.0,
        high_target_voltage_offset=0.5,
        low_target_voltage_offset=-1.0,
        cue_memory_mode="synaptic_eligibility",
        cue_memory_signal="spike",
        cue_memory_tau_ms=100.0,
        cue_memory_gain=1.0,
        cue_memory_floor=0.0,
        surfaces=("<recurrent>",),
        edge_chunk_size=1,
    )
    rule.build(rnn)

    coincident = rule.compute_updates(observations)[0][1]
    no_post_cue = replace(
        observations,
        spikes=tf.constant([[[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
    )

    assert float(coincident.numpy()[0]) < 0.0
    np.testing.assert_allclose(rule.compute_updates(no_post_cue)[0][1].numpy(), 0.0)


def test_neuron_local_target_pool_scope_excludes_non_target_readout():
    rnn, _, _ = _fake_rnn()
    observations = replace(
        _observations(),
        spikes=tf.zeros([1, 3, 2], dtype=tf.float32),
        voltages=tf.ones([1, 3, 2], dtype=tf.float32),
        targets={
            "class_label": tf.constant([0], dtype=tf.int32),
            "delay_ms": tf.constant([0.0], dtype=tf.float32),
        },
    )
    rule = NeuronLocalThreeFactorLearningRule(
        modulator="postsynaptic_voltage_error",
        pool_a_start=0,
        pool_a_end=1,
        pool_b_start=1,
        pool_b_end=2,
        cue_duration_ms=1.0,
        response_window_ms=2.0,
        high_target_voltage_offset=0.5,
        low_target_voltage_offset=-1.0,
        target_scope="target_pool_only",
        surfaces=("<recurrent>",),
    )
    rule.build(rnn)

    local_factor = rule._local_factor(observations, tf.ones_like(observations.spikes))

    assert np.all(local_factor.numpy()[:, 1:, 0] < 0.0)
    np.testing.assert_allclose(local_factor.numpy()[:, :, 1], 0.0)


def test_neuron_local_update_step_restores_from_checkpoint(tmp_path):
    rule = NeuronLocalThreeFactorLearningRule()
    rule.on_global_step_end()
    checkpoint_path = tf.train.Checkpoint(rule=rule).save(
        str(tmp_path / "neuron_local_rule")
    )
    restored = NeuronLocalThreeFactorLearningRule()

    tf.train.Checkpoint(rule=restored).restore(checkpoint_path).assert_consumed()

    assert int(restored.update_step.numpy()) == 1


def test_custom_learning_rule_registration():
    @register_learning_rule(name="test_custom_rule")
    class CustomRule(LearningRule):
        @classmethod
        def module(cls):
            return "unused_name"

        def _build_weight_surfaces(self, rnn):
            return ()

        def compute_updates(self, observations):
            return ()

    assert LearningRules().get_rule("test_custom_rule") is CustomRule


def test_training_engine_passes_local_and_direct_learning_signals():
    class RecordingRule(LearningRule):
        @classmethod
        def module(cls):
            return "recording"

        def _build_weight_surfaces(self, rnn):
            return ()

        def compute_updates(self, observations):
            self.observations = observations
            return ()

        def apply_updates(self, optimizer, updates):
            self.applied = True

    spikes = tf.constant([[[0.0, 1.0], [1.0, 0.0]]])
    voltages = tf.ones_like(spikes)
    direct_weight = tf.keras.layers.Layer().add_weight(
        shape=(1,), initializer=tf.keras.initializers.Constant(3.0)
    )
    rnn = SimpleNamespace(
        run_extractor=lambda inputs, state: ((spikes, voltages), tf.zeros([1, 2])),
        model=SimpleNamespace(trainable_variables=[direct_weight]),
    )
    engine = TrainingEngine(rnn=rnn, n_epochs=1, steps_per_epoch=1)
    parameter = engine.add_parameters("default", batch_size=1, seq_len=2)
    parameter.add_loss_function(
        "combined",
        lambda spikes, voltages, **kwargs: (
            tf.reduce_sum(tf.square(spikes))
            + 2.0 * tf.reduce_sum(voltages)
            + tf.reduce_sum(tf.square(direct_weight))
        ),
    )
    rule = RecordingRule()
    engine.set_learning_rule(rule)
    rule.weight_surfaces = (
        WeightSurface(
            name="<recurrent>",
            variable=direct_weight,
            indices=tf.constant([[0, 0]], dtype=tf.int64),
            synapse_types=tf.constant([0], dtype=tf.int64),
            source="recurrent",
        ),
    )
    engine._normalizers = None

    targets = {"class_label": tf.constant([1], dtype=tf.int32)}
    loss_values = engine._train_step_local_single(
        [tf.zeros([1, 2, 0])], targets, (tf.zeros([1, 2]),)
    )

    np.testing.assert_allclose(
        rule.observations.spike_learning_signal.numpy(), 2.0 * spikes.numpy()
    )
    np.testing.assert_allclose(rule.observations.voltage_learning_signal.numpy(), 2.0)
    np.testing.assert_allclose(
        rule.observations.direct_weight_gradients[0].numpy(), [6.0]
    )
    assert rule.observations.targets is targets
    assert rule.applied
    assert float(loss_values["__total_loss"].numpy()) == 19.0


def test_loss_scaling_uses_physical_values_for_local_rule():
    variable = tf.Variable([1.0])
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.SGD(learning_rate=0.1)
    )
    optimizer.build([variable])

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(variable))
        scaled_loss = optimizers.scale_loss_for_optimizer(optimizer, loss)
    scaled_gradient = tape.gradient(scaled_loss, variable)

    unscaled = optimizers.unscale_gradients_for_local_rule(optimizer, [scaled_gradient])
    prepared = optimizers.prepare_local_gradients_for_optimizer(optimizer, unscaled)
    optimizer.apply_gradients(zip(prepared, [variable]))

    np.testing.assert_allclose(unscaled[0].numpy(), [2.0])
    np.testing.assert_allclose(variable.numpy(), [0.8])
