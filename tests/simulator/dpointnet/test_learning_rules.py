from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')

from bmtk.simulator.dpointnet.learning_rules import (
    EPropLearningRule,
    LearningRuleObservations,
    LearningRules,
    ModPropLearningRule,
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
            'lgn': {
                'input_dim': 1,
                'input_type': 'spikes',
                'input_weight_values': input_weight,
                'input_indices': tf.constant([[1, 0]], dtype=tf.int64),
                'input_syn_ids': tf.constant([0], dtype=tf.int64),
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
        spike_learning_signal=tf.constant(
            [[[2.0, 1.0], [3.0, 4.0], [4.0, 5.0]]]
        ),
        voltage_learning_signal=tf.zeros([1, 3, 2], dtype=tf.float32),
        direct_weight_gradients=(tf.zeros([1]), tf.zeros([1])),
    )


def _modprop_observations():
    observations = _observations()
    return replace(
        observations,
        input_spikes=tf.concat(
            [observations.input_spikes, tf.zeros([1, 1, 1])], axis=1
        ),
        spikes=tf.concat(
            [observations.spikes, tf.zeros([1, 1, 2])], axis=1
        ),
        voltages=tf.concat(
            [observations.voltages, tf.ones([1, 1, 2])], axis=1
        ),
        spike_learning_signal=tf.concat(
            [
                observations.spike_learning_signal,
                tf.constant([[[2.0, 0.0]]]),
            ],
            axis=1,
        ),
        voltage_learning_signal=tf.zeros([1, 4, 2]),
    )


def test_eprop_computes_delayed_recurrent_and_input_updates():
    rnn, recurrent, input_weight = _fake_rnn()
    rule = EPropLearningRule(
        edge_chunk_size=1,
        surfaces=('<recurrent>', 'lgn'),
    )
    rule.build(rnn)

    updates = {surface.name: gradient for surface, gradient in rule.compute_updates(_observations())}

    np.testing.assert_allclose(updates['<recurrent>'].numpy(), [2.0])
    np.testing.assert_allclose(updates['lgn'].numpy(), [5.0])
    assert rule.weight_surfaces[0].variable is recurrent
    assert rule.weight_surfaces[1].variable is input_weight


def test_eprop_applies_bounds_and_tracks_serializable_step(tmp_path):
    variable = tf.Variable([0.9], dtype=tf.float32, trainable=True)
    surface = WeightSurface(
        name='<recurrent>',
        variable=variable,
        indices=tf.constant([[0, 0]], dtype=tf.int64),
        synapse_types=tf.constant([0], dtype=tf.int64),
        source='recurrent',
    )
    rule = EPropLearningRule(min_weight=-0.5, max_weight=0.5)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    rule.apply_updates(optimizer, ((surface, tf.constant([-1.0])),))
    rule.on_global_step_end()

    np.testing.assert_allclose(variable.numpy(), [0.5])
    assert int(rule.update_step.numpy()) == 1
    checkpoint_path = tf.train.Checkpoint(rule=rule).save(str(tmp_path / 'rule'))

    restored = EPropLearningRule()
    tf.train.Checkpoint(rule=restored).restore(checkpoint_path).assert_consumed()
    assert int(restored.update_step.numpy()) == 1


def test_eprop_config_round_trip():
    rule = EPropLearningRule(
        edge_chunk_size=64,
        surfaces=('lgn',),
        learning_signal_clip=2.0,
        gradient_clip_norm=3.0,
    )
    config = rule.get_config()
    name = config.pop('name')
    restored = LearningRules().get_rule(name)(**config)

    assert restored.get_config() == rule.get_config()


def test_surface_validation_rejects_each_irrelevant_request():
    rnn, _, _ = _fake_rnn()
    with pytest.raises(ValueError, match='Unknown learning-rule surface.*missing'):
        EPropLearningRule(surfaces=('<recurrent>', 'missing')).build(rnn)

    rnn._cell.inputs['lgn']['input_weight_values'] = tf.Variable(
        [2.0], trainable=False
    )
    with pytest.raises(ValueError, match='not trainable: lgn'):
        EPropLearningRule(surfaces=('lgn',)).build(rnn)

    rnn._cell.inputs['lgn']['input_weight_values'] = tf.Variable(
        [2.0], trainable=True
    )
    rnn._cell.inputs['lgn']['input_type'] = 'poisson_spikes_internal'
    with pytest.raises(ValueError, match='cannot observe internally generated'):
        EPropLearningRule(surfaces=('lgn',)).build(rnn)


def test_three_factor_selects_spike_or_voltage_modulator():
    rnn, _, _ = _fake_rnn()
    observations = _observations()
    spike_rule = ThreeFactorLearningRule(
        signal='spike', edge_chunk_size=1, surfaces=('<recurrent>',)
    )
    spike_rule.build(rnn)
    voltage_rule = ThreeFactorLearningRule(
        signal='voltage', edge_chunk_size=1, surfaces=('<recurrent>',)
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
    with pytest.raises(ValueError, match='Unknown three-factor signal'):
        ThreeFactorLearningRule(signal='irrelevant')


def test_modprop_adds_delayed_type_specific_modulation():
    rnn, _, _ = _fake_rnn()
    filters = np.zeros((2, 2, 1), dtype=np.float32)
    filters[0, 0, 0] = 2.0
    filters[0, 1, 0] = 3.0
    observations = _modprop_observations()

    eprop = EPropLearningRule(
        edge_chunk_size=1, surfaces=('<recurrent>', 'lgn')
    )
    eprop.build(rnn)
    modprop = ModPropLearningRule(
        modulatory_filters=filters,
        edge_chunk_size=1,
        surfaces=('<recurrent>', 'lgn'),
    )
    modprop.build(rnn)
    eprop_updates = {
        surface.name: gradient
        for surface, gradient in eprop.compute_updates(observations)
    }
    modprop_updates = {
        surface.name: gradient
        for surface, gradient in modprop.compute_updates(observations)
    }

    np.testing.assert_allclose(
        modprop_updates['<recurrent>'] - eprop_updates['<recurrent>'], [1.0]
    )
    np.testing.assert_allclose(
        modprop_updates['lgn'] - eprop_updates['lgn'], [3.0]
    )


def test_modprop_supports_configurable_batch_size():
    rnn, _, _ = _fake_rnn()
    observations = _modprop_observations()
    batch_size = 3
    observations = replace(
        observations,
        input_spikes=tf.repeat(observations.input_spikes, batch_size, axis=0),
        spikes=tf.repeat(observations.spikes, batch_size, axis=0),
        voltages=tf.repeat(observations.voltages, batch_size, axis=0),
        initial_state=(
            tf.repeat(observations.initial_state[0], batch_size, axis=0),
        ),
        spike_learning_signal=tf.repeat(
            observations.spike_learning_signal, batch_size, axis=0
        ),
        voltage_learning_signal=tf.repeat(
            observations.voltage_learning_signal, batch_size, axis=0
        ),
    )
    rule = ModPropLearningRule(
        modulatory_filters=np.ones((2, 2, 2), dtype=np.float32),
        edge_chunk_size=1,
        surfaces=('<recurrent>',),
    )
    rule.build(rnn)

    gradient = rule.compute_updates(observations)[0][1]

    assert gradient.shape == (1,)
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)))


def test_modprop_builds_fixed_filters_from_initial_type_weights():
    rnn, _, _ = _fake_rnn()
    rule = ModPropLearningRule(
        filter_taps=2,
        mean_activity=0.5,
        surfaces=('<recurrent>',),
    )
    rule.build(rnn)

    np.testing.assert_array_equal(rule.node_type_values.numpy(), [10, 20])
    np.testing.assert_allclose(
        rule.modulatory_filters.numpy(),
        np.array([[[0.0, 0.0], [1.0, 0.0]],
                  [[0.0, 0.0], [0.0, 0.0]]]),
    )
    assert not rule.modulatory_filters.trainable

    config = rule.get_config()
    rnn._cell.recurrent_weight_values.assign([9.0])
    name = config.pop('name')
    rebuilt = LearningRules().get_rule(name)(**config)
    rebuilt.build(rnn)
    np.testing.assert_allclose(
        rebuilt.modulatory_filters.numpy(),
        rule.modulatory_filters.numpy(),
    )


def test_modprop_rejects_irrelevant_or_inconsistent_filter_options():
    filters = np.zeros((2, 2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match='mean_activity is irrelevant'):
        ModPropLearningRule(
            mean_activity=0.5, modulatory_filters=filters
        )
    with pytest.raises(ValueError, match='provides 1 taps'):
        rnn, _, _ = _fake_rnn()
        rule = ModPropLearningRule(
            filter_taps=2,
            modulatory_filters=filters,
            surfaces=('<recurrent>',),
        )
        rule.build(rnn)


def test_modprop_config_and_filter_checkpoint_round_trip(tmp_path):
    filters = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    rnn, _, _ = _fake_rnn()
    rule = ModPropLearningRule(
        modulatory_filters=filters,
        edge_chunk_size=4,
        surfaces=('<recurrent>',),
    )
    rule.build(rnn)
    config = rule.get_config()
    name = config.pop('name')
    rebuilt = LearningRules().get_rule(name)(**config)
    rebuilt.build(rnn)
    assert rebuilt.get_config() == rule.get_config()

    rule.modulatory_filters.assign(tf.zeros_like(rule.modulatory_filters))
    checkpoint_path = tf.train.Checkpoint(rule=rule).save(
        str(tmp_path / 'modprop')
    )
    rebuilt.modulatory_filters.assign(tf.ones_like(rebuilt.modulatory_filters))
    tf.train.Checkpoint(rule=rebuilt).restore(checkpoint_path).assert_consumed()
    np.testing.assert_allclose(rebuilt.modulatory_filters.numpy(), 0.0)


def test_three_factor_config_round_trip():
    rule = ThreeFactorLearningRule(
        signal='voltage',
        spike_signal_scale=2.0,
        voltage_signal_scale=3.0,
        surfaces=('<recurrent>',),
    )
    config = rule.get_config()
    name = config.pop('name')
    rebuilt = LearningRules().get_rule(name)(**config)

    assert rebuilt.get_config() == rule.get_config()


def test_custom_learning_rule_registration():
    @register_learning_rule(name='test_custom_rule')
    class CustomRule(LearningRule):
        @classmethod
        def module(cls):
            return 'unused_name'

        def _build_weight_surfaces(self, rnn):
            return ()

        def compute_updates(self, observations):
            return ()

    assert LearningRules().get_rule('test_custom_rule') is CustomRule


def test_training_engine_passes_local_and_direct_learning_signals():
    class RecordingRule(LearningRule):
        @classmethod
        def module(cls):
            return 'recording'

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
    parameter = engine.add_parameters('default', batch_size=1, seq_len=2)
    parameter.add_loss_function(
        'combined',
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
            name='<recurrent>',
            variable=direct_weight,
            indices=tf.constant([[0, 0]], dtype=tf.int64),
            synapse_types=tf.constant([0], dtype=tf.int64),
            source='recurrent',
        ),
    )
    engine._normalizers = None

    loss_values = engine._train_step_local_single(
        [tf.zeros([1, 2, 0])], None, (tf.zeros([1, 2]),)
    )

    np.testing.assert_allclose(
        rule.observations.spike_learning_signal.numpy(), 2.0 * spikes.numpy()
    )
    np.testing.assert_allclose(
        rule.observations.voltage_learning_signal.numpy(), 2.0
    )
    np.testing.assert_allclose(
        rule.observations.direct_weight_gradients[0].numpy(), [6.0]
    )
    assert rule.applied
    assert float(loss_values['__total_loss'].numpy()) == 19.0


def test_keras3_loss_scaling_uses_physical_values_for_local_rule():
    variable = tf.Variable([1.0])
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.SGD()
    )
    optimizer.build([variable])
    scale = tf.cast(optimizer.dynamic_scale, tf.float32)
    scaled_gradient = tf.constant([2.0]) * scale

    unscaled = optimizers.unscale_gradients_for_local_rule(
        optimizer, [scaled_gradient]
    )
    prepared = optimizers.prepare_local_gradients_for_optimizer(
        optimizer, unscaled
    )

    np.testing.assert_allclose(unscaled[0].numpy(), [2.0])
    np.testing.assert_allclose(prepared[0].numpy(), scaled_gradient.numpy())
