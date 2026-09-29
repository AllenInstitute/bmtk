import copy
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import LossModules, VoltageRateFloor
from bmtk.simulator.dpointnet.loss_functions.voltage_rate_floor import voltage_rate_floor_step
from bmtk.simulator.dpointnet.optimizers import ExponentiatedAdam
from bmtk.simulator.dpointnet.rnn_model import RNN
from bmtk.simulator.dpointnet.training import TrainingEngine
from test_nest_dynamics import make_network_inputs


def make_rnn(mode="nest", compact=True, checkpointing=False, dt=1.0, policy="float32",
             enabled=True, strategy=None, initialize_rates=None, selective=False,
             replay_mode="record"):
    network, inputs = make_network_inputs()
    rnn = RNN(
        seq_len=6, batch_size=2, dt=dt, dtype=policy,
        cell_params=dict(dynamics_mode=mode, tau_basis=[2.0], hard_reset=False,
                         dt=dt, track_voltage_penalty=compact,
                         return_voltage_sequences=not compact,
                         state_precision="selective" if selective else "compute",
                         temporal_gradient_precision="float32" if selective else "compute",
                         current_replay_mode=replay_mode if selective else None,
                         temporal_checkpoint_chunk_size=2),
    )
    rnn.strategy = strategy or tf.distribute.get_strategy()
    rnn._recurrent_networks["test"] = SimpleNamespace(to_dict=lambda: copy.deepcopy(network))
    rnn._input_networks["drive"] = SimpleNamespace(
        name="drive", n_spiking_nodes=1, to_dict=lambda: copy.deepcopy(inputs["drive"])
    )
    engine = rnn.set_training(rnn=rnn, n_epochs=1, steps_per_epoch=1,
                              gradient_checkpointing=checkpointing,
                              gradient_checkpoint_chunk_size=2)
    parameter = engine.add_parameters("test", batch_size=2, seq_len=6)
    loss = VoltageRateFloor(rnn) if enabled else None
    if enabled:
        if initialize_rates is not None:
            loss.initialize_rates(initialize_rates)
        parameter.add_loss_function("floor", loss)
    with rnn.strategy.scope():
        engine.set_optimizer(tf.keras.optimizers.SGD(0.01))
    rnn.build()
    # Build the extractor without advancing a noise stream.
    rnn.extractor_model = rnn._build_extractor_model()
    engine.prepare_gradient_checkpointing()
    return rnn, engine, loss


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_independent_value_gradient_gate_and_denominator(dtype):
    values = np.array([[0.1, 0.5, 1.2, -0.2], [0.0, 0.3, 0.7, 0.1]], np.float32)
    active = np.array([[1, 0, 1, 1], [1, 1, 0, 1]], np.float32)
    gate_values = np.array([0.0, 1.0, 0.5, 0.25], np.float32)
    voltage = tf.Variable(values, dtype=dtype)
    gate = tf.Variable(gate_values)
    with tf.GradientTape() as tape:
        actual = tf.reduce_mean(voltage_rate_floor_step(voltage, active, gate, cost=2.0))
    dv, dg = tape.gradient(actual, [voltage, gate])
    rounded = np.asarray(voltage.numpy(), np.float32)
    deficit = np.maximum(0.9 - rounded, 0)
    np.testing.assert_allclose(actual, np.mean(2 * deficit**2 * active * gate_values), rtol=1e-6)
    expected_gradient = (-4 * deficit * active * gate_values / values.size).astype(dtype.as_numpy_dtype)
    np.testing.assert_array_equal(dv, expected_gradient)
    assert dg is None


@pytest.mark.parametrize("mode", ["nest", "legacy"])
@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_bootstrap_replay_validation_and_checkpoint(mode, compact, checkpointing, tmp_path):
    rnn, engine, loss = make_rnn(mode, compact, checkpointing)
    x = tf.ones((2, 6, 1))
    initial = rnn.cell.zero_state(2, tf.float32)
    np.testing.assert_array_equal(loss.gate, [0])
    first = engine._train_step_single([x], {}, initial)
    assert first["test"]["floor"].numpy() == 0
    assert loss.accepted_updates.numpy() == 1
    assert loss.initialized.numpy() == 1
    np.testing.assert_array_equal(loss.gate, [1])
    second = engine._train_step_single([x], {}, initial)
    assert second["test"]["floor"].numpy() > 0
    assert loss.accepted_updates.numpy() == 2
    state = [np.array(v) for v in (loss.rate_ema_hz, loss.gate, loss.accepted_updates)]
    engine._validation_step([x], [{}], initial, "single")
    for actual, expected in zip((loss.rate_ema_hz, loss.gate, loss.accepted_updates), state):
        np.testing.assert_array_equal(actual, expected)
    checkpoint = tf.train.Checkpoint(model=rnn.model, optimizer=engine.optimizer)
    path = checkpoint.save(str(tmp_path / "native"))
    loss.initialize_rates([7.0])
    loss.cost.assign(9.0)
    checkpoint.restore(path).assert_consumed()
    np.testing.assert_array_equal(loss.gate, state[1])
    assert loss.cost.numpy() == 1.0
    rnn.cleanup()


@pytest.mark.parametrize("mode", ["nest", "legacy"])
@pytest.mark.parametrize("policy", ["float32", "float16"])
def test_full_bptt_and_exact_recompute_gradients_and_disabled_parity(mode, policy):
    enabled, engine, loss = make_rnn(mode=mode, policy=policy)
    disabled, _, _ = make_rnn(mode=mode, policy=policy, enabled=False)
    loss.initialize_rates([0.0])
    x = tf.ones((2, 6, 1), dtype=enabled.dtype)
    initial = enabled.cell.zero_state(2, enabled.dtype)
    baseline = disabled.run_extractor(x, disabled.cell.zero_state(2, disabled.dtype))

    def value_gradient():
        with tf.GradientTape() as tape:
            output = engine._run_extractor(x, initial)
            value = loss(output[0][0], output[1:])
        return value, output, tape.gradient(value, enabled.model.trainable_variables)

    full_value, full_output, full_gradient = value_gradient()
    for actual, expected in zip(tf.nest.flatten(full_output)[:-1], tf.nest.flatten(baseline)):
        np.testing.assert_array_equal(actual, expected)
    assert full_output[0][0].dtype == enabled.dtype
    assert full_output[-1].dtype == tf.float32
    assert any(np.any(g.numpy() != 0) for g in full_gradient if g is not None)
    engine.gradient_checkpointing = True
    engine.prepare_gradient_checkpointing()
    replay_value, replay_output, replay_gradient = value_gradient()
    np.testing.assert_allclose(replay_value, full_value, rtol=1e-6)
    for actual, expected in zip(replay_gradient, full_gradient):
        if expected is not None:
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-6)
    assert loss.accepted_updates.numpy() == 0
    enabled.cleanup()
    disabled.cleanup()


def test_native_compiled_mirrored_strategy_step():
    strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])
    rnn, engine, loss = make_rnn(strategy=strategy, checkpointing=True)
    x = tf.ones((2, 6, 1))
    initial = rnn.cell.zero_state(2, rnn.dtype)
    result = engine._distributed_train_step([x], {}, initial)
    assert result["test"]["floor"].numpy() == 0
    assert loss.accepted_updates.numpy() == 1
    result = engine._distributed_train_step([x], {}, initial)
    assert result["test"]["floor"].numpy() > 0
    assert loss.accepted_updates.numpy() == 2
    rnn.cleanup()


@pytest.mark.parametrize("approach", ["parallel", "series", "series_accumulate"])
def test_condition_history_pooling_and_update_count(approach):
    rnn, engine, loss = make_rnn()
    engine._parameters = [
        SimpleNamespace(name="a", batch_size=1, loss_functions={"floor": loss}),
        SimpleNamespace(name="b", batch_size=3, loss_functions={"floor": loss}),
    ]
    engine._batch_indices = None
    engine._inputs_sig_factory = SimpleNamespace(build=lambda ys: ys, lu_tables=[{}, {}])
    weight = rnn.model.trainable_variables[0]
    def forward(x, initial):
        accumulator = tf.reduce_sum(x, axis=(1, 2))[:, None] + 0.0 * tf.reduce_sum(weight)
        return (x, x), accumulator
    engine._run_extractor = forward
    xs = [tf.zeros((1, 6, 1)), tf.ones((3, 6, 1))]
    if approach == "parallel":
        result = engine._train_step_parallel(xs, [{}, {}], None)
        np.testing.assert_allclose(loss.rate_ema_hz, [750.0])
        np.testing.assert_allclose(result["a"]["floor"] + result["b"]["floor"], 1.5)
        np.testing.assert_allclose(result["__total_loss"], .75)
        assert loss.accepted_updates.numpy() == 1
    elif approach == "series":
        engine._train_step_series(xs, [{}, {}], None)
        np.testing.assert_allclose(loss.rate_ema_hz, [50.0], rtol=1e-6)
        assert loss.accepted_updates.numpy() == 2
    else:
        result = engine._distributed_train_step_series_accumulate(xs, [{}, {}], None)
        np.testing.assert_allclose(result["__total_loss"], .75)
        rnn.run_extractor = forward
        validation = engine._validation_step(xs, [{}, {}], None, "series_accumulate")
        np.testing.assert_allclose(validation["__total_loss"], result["__total_loss"])
        np.testing.assert_allclose(loss.rate_ema_hz, [750.0])
        assert loss.accepted_updates.numpy() == 1
    rnn.cleanup()


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("unroll", [False, True])
def test_fp32_accumulator_survives_mixed_precision_layer_boundaries(mode, unroll):
    rnn, _, loss = make_rnn(mode=mode, policy="float16", initialize_rates=[1.0])
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    state = list(rnn.cell.zero_state(2, rnn.dtype))
    state[-1] = tf.constant([[65536.0], [65537.0]], tf.float32)
    inputs = tf.zeros((2, 6, 1), dtype=rnn.dtype)
    layer = ExplicitStateRNN(rnn.cell, return_sequences=True, return_state=True, unroll=unroll)
    layer._autocast = False
    layer.autocast = False
    result = layer(inputs, initial_state=state)
    assert result[-1].dtype == tf.float32
    np.testing.assert_array_equal(result[-1], state[-1])
    direct = rnn.cell(inputs[:, 0], tuple(state))
    np.testing.assert_array_equal(direct[1][-1], state[-1])
    rnn.cleanup()


@pytest.mark.parametrize("optimizer_cls", [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam, ExponentiatedAdam])
@pytest.mark.parametrize("scaled", [False, True])
def test_commit_once_and_rejected_dynamic_scale(optimizer_cls, scaled):
    rnn, engine, loss = make_rnn()
    optimizer = optimizer_cls(learning_rate=0.01)
    if scaled:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    engine.set_optimizer(optimizer)
    loss.initialize_rates([0.05])
    variables = rnn.model.trainable_variables

    @tf.function
    def apply(value):
        return engine._apply_gradients([tf.ones_like(v) * value for v in variables],
                                       (tf.constant([1.0]), tf.constant(1000.0)))

    apply(tf.constant(1.0))
    assert loss.accepted_updates.numpy() == 1
    np.testing.assert_allclose(loss.rate_ema_hz, [.95 * .05 + .05 * 1.0], atol=1e-7)
    if scaled:
        before = loss.rate_ema_hz.numpy().copy()
        apply(tf.constant(float("inf")))
        assert loss.accepted_updates.numpy() == 1
        np.testing.assert_array_equal(loss.rate_ema_hz, before)
    rnn.cleanup()


def test_configuration_and_measured_initialization():
    assert LossModules().get_module("VoltageRateFloor") is VoltageRateFloor
    for kwargs in (dict(cost=-1), dict(floor_hz=0), dict(ema_decay=1),
                   dict(target=float("nan")), dict(neuron_ids=[0])):
        with pytest.raises(ValueError):
            VoltageRateFloor(SimpleNamespace(dt=1.0), **kwargs)
    rnn, engine, loss = make_rnn(dt=.5)
    loss.initialize_rates([.2])
    assert loss.gate.numpy()[0] == 0
    counts, samples = engine._voltage_floor_rate_statistics(tf.ones((2, 6, 1)))
    engine._apply_gradients([tf.zeros_like(v) for v in rnn.model.trainable_variables], (counts, samples))
    np.testing.assert_allclose(loss.rate_ema_hz, [.95 * .2 + .05 * 2000], rtol=1e-6)
    with pytest.raises(ValueError, match="before building"):
        VoltageRateFloor(rnn)
    rnn.cleanup()


@pytest.mark.parametrize("mode", ["nest", "legacy"])
@pytest.mark.parametrize("refractory", [0, 3])
def test_pre_reset_voltage_and_native_refractory_mask(mode, refractory):
    rnn, _, loss = make_rnn(mode=mode, initialize_rates=[0.0])
    loss.target.assign(1.5)
    voltage = tf.Variable([[1.2], [1.2]])
    state = list(rnn.cell.zero_state(2, tf.float32))
    state[1] = voltage
    state[2] = tf.fill(tf.shape(state[2]), tf.cast(refractory, state[2].dtype))
    with tf.GradientTape() as tape:
        _, final = rnn.cell(tf.zeros((2, 1)), tuple(state))
        actual = tf.reduce_mean(final[-1])
    gradient = tape.gradient(actual, voltage)
    if refractory:
        np.testing.assert_array_equal(actual, 0)
        np.testing.assert_array_equal(gradient, tf.zeros_like(voltage))
    else:
        expected_voltage = voltage.numpy() * rnn.cell.decay.numpy()
        np.testing.assert_allclose(actual, np.mean((1.5 - expected_voltage)**2), rtol=1e-6)
        retention = 1.0 - rnn.cell._voltage_gradient_dampening
        np.testing.assert_allclose(gradient, -retention * (1.5 - expected_voltage) * rnn.cell.decay.numpy(),
                                   rtol=1e-6)
        if mode == "nest":
            assert np.max(final[1].numpy()) < 1.0  # Existing post-reset state remains unchanged.
    assert loss.accepted_updates.numpy() == 0
    rnn.cleanup()


def test_model_weight_save_restores_history_and_config(tmp_path):
    rnn, _, loss = make_rnn(initialize_rates=[0.025])
    path = str(tmp_path / "floor.weights.h5")
    rnn.model.save_weights(path)
    loss.initialize_rates([9.0])
    loss.target.assign(7.0)
    rnn.model.load_weights(path)
    np.testing.assert_allclose(loss.rate_ema_hz, [.025])
    np.testing.assert_allclose(loss.gate, [.75])
    np.testing.assert_allclose(loss.target, .9)
    rnn.cleanup()


def test_native_json_registration_precedes_building_losses(monkeypatch):
    from bmtk.simulator.core.simulation_config import SimulationConfig
    from bmtk.simulator.dpointnet.network_adaptor import NetworkAdaptor

    network, inputs = make_network_inputs()
    rec = SimpleNamespace(name="test", to_dict=lambda: copy.deepcopy(network))
    drive = SimpleNamespace(name="drive", n_spiking_nodes=1,
                            to_dict=lambda: copy.deepcopy(inputs["drive"]))
    monkeypatch.setattr(NetworkAdaptor, "from_dict", lambda _: ([rec], [drive]))

    def add_network(rnn, net):
        if net is rec:
            rnn._recurrent_networks[net.name] = net
        else:
            rnn._input_networks[net.name] = net
            rnn._inputs_order.append(net.name)

    monkeypatch.setattr(RNN, "add_network", add_network)
    losses = {
        "range": {"module": "VoltageRegularization", "online": True},
        "floor": {"module": "VoltageRateFloor"},
        "disabled": {"module": "VoltageRateFloor", "enabled": False, "cost": 9},
    }
    config = SimulationConfig({
        "run": {"seq_len": 6, "batch_size": 2, "dt": 1.0},
        "networks": {}, "components": {}, "inputs": {},
        "rnn_cell_params": {"cell_model": "GLIF3", "tau_basis": [2.0],
                            "track_voltage_penalty": True, "return_voltage_sequences": False},
        "training": {
            "n_epochs": 1, "steps_per_epoch": 1, "training_approach": "parallel",
            "learning_rate": .01, "optimizer": {"name": "sgd"},
            "parameters": [
                {"name": name, "batch_size": 1, "inputs": {}, "loss_functions": copy.deepcopy(losses)}
                for name in ("a", "b")
            ],
        },
    })
    rnn = RNN.from_config(config)
    first, second = rnn.training_engine.parameters
    assert first.loss_functions["floor"] is second.loss_functions["floor"]
    assert len(rnn._online_voltage_losses) == 1
    assert rnn._model_built
    assert rnn.cell.zero_state(2, rnn.dtype)[-1].shape == (2, 1)
    rnn.cleanup()


@pytest.mark.parametrize("mode", ["nest", "legacy"])
@pytest.mark.parametrize("extension", ["npz", "pkl"])
@pytest.mark.parametrize("omit_noise", [False, True])
def test_physical_initial_state_cache_can_omit_accumulator(mode, extension, omit_noise, tmp_path):
    import pickle
    from bmtk.simulator.dpointnet.state_modules.cached_states import CachedInitState

    rnn, _, loss = make_rnn(mode=mode, dt=.5 if mode == "nest" else 1.0)
    state, names = rnn.cell.zero_state(2, tf.float32, with_names=True)
    physical = [(name, value.numpy()) for name, value in zip(names[:-1], state[:-1])
                if not omit_noise or name != "noise_step0"]
    path = tmp_path / ("state." + extension)
    if extension == "npz":
        np.savez(path, **dict(physical))
    else:
        with path.open("wb") as stream:
            pickle.dump(tuple(value for _, value in physical), stream)
    restored = CachedInitState(str(path), rnn=rnn).get_state()
    for actual, expected in zip(restored, state):
        np.testing.assert_array_equal(actual, expected)
    assert loss.initialized.numpy() == 0
    rnn.cleanup()


@pytest.mark.parametrize("as_list", [False, True])
def test_randomized_initial_state_defaults_online_channel(as_list):
    from bmtk.simulator.dpointnet.state_modules.randomized_state import RandomizedStateModule

    rnn, _, _ = make_rnn()
    _, names = rnn.cell.zero_state(2, tf.float32, with_names=True)
    params = {name: dict(random_func="const", value=0) for name in names[:-1]
              if name != "noise_step0"}
    module = RandomizedStateModule(rnn, list(params.values()) if as_list else params)
    actual = module.get_state()[-1]
    assert actual.shape == (2, 1)
    assert actual.dtype == tf.float32
    np.testing.assert_array_equal(actual, [[0.0], [0.0]])
    rnn.cleanup()
