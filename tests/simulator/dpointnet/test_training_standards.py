import copy
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet._options import (
    UNSET,
    resolve_precision_options,
    resolve_renamed_option,
)
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import fused_nest_state_available
from bmtk.simulator.dpointnet.rnn_model import RNN
from bmtk.simulator.dpointnet.migration import preserve_previous_defaults
from bmtk.simulator.dpointnet.training import TrainingEngine
from test_precision_credit import make_cell


@pytest.fixture(autouse=True)
def restore_precision():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("switches,dtype,state,temporal", [
    ({}, "float16", "selective", "compute"),
    ({"mixed_precision": False}, "float32", "compute", "compute"),
    ({"fp32_voltage_and_asc": False}, "float16", "compute", "compute"),
    ({"fp32_temporal_gradients": True}, "float16", "selective", "float32"),
    ({"mixed_precision": False, "fp32_temporal_gradients": True},
     "float32", "compute", "compute"),
])
def test_precision_switch_resolution(switches, dtype, state, temporal):
    actual, options, report = resolve_precision_options(UNSET, switches, {})
    assert actual == dtype
    assert options == {
        "state_precision": state,
        "temporal_gradient_precision": temporal,
    }
    assert report["requested_switches"] == switches


@pytest.mark.parametrize("dtype", ["float16", "float32", "bfloat16", "float64"])
def test_explicit_legacy_precision_is_authoritative(dtype):
    original = {"state_precision": "compute", "temporal_gradient_precision": "compute"}
    resolved, options, _ = resolve_precision_options(dtype, None, original)
    assert resolved == dtype
    assert options == original
    assert original == {"state_precision": "compute", "temporal_gradient_precision": "compute"}


@pytest.mark.parametrize("dtype,switches,options,match", [
    ("float32", {"mixed_precision": True}, {}, "dtype conflicts"),
    (UNSET, {"fp32_voltage_and_asc": False}, {"state_precision": "selective"}, "state_precision conflicts"),
    (UNSET, {"fp32_temporal_gradients": True}, {"temporal_gradient_precision": "compute"}, "temporal_gradient_precision conflicts"),
    (UNSET, {"fp32_temporal_gradients": True, "fp32_voltage_and_asc": False}, {}, "require fp32_voltage"),
    (UNSET, {"mixed_precision": 1}, {}, "must be true or false"),
    (UNSET, {"unknown": True}, {}, "Unknown precision"),
])
def test_precision_errors_are_explicit(dtype, switches, options, match):
    with pytest.raises(ValueError, match=match):
        resolve_precision_options(dtype, switches, options)


def test_rnn_and_training_omitted_defaults():
    rnn = RNN(batch_size=2, seq_len=5)
    assert rnn.dtype == tf.float16
    assert rnn.cell_params["state_precision"] == "selective"
    assert rnn.cell_params["acceleration_profile"] == "auto"
    engine = TrainingEngine(rnn, n_epochs=1, steps_per_epoch=1)
    assert engine.gradient_checkpointing is True
    assert engine.gradient_checkpoint_chunk_size == 5
    assert rnn.batch_size == 2
    assert rnn.seq_len == 5
    assert rnn._resolve_cell_params(training=True)["hard_reset"] is True
    assert rnn._resolve_cell_params(training=True)["hard_reset_gradient_mode"] == "soft_surrogate"


@pytest.mark.parametrize("sequence_length", [1, 5, 25, 500])
def test_omitted_checkpoint_size_tracks_effective_training_length(sequence_length):
    rnn = RNN(batch_size=2, seq_len=100)
    engine = TrainingEngine(rnn, 1, 1, training_approach="single")
    assert engine.gradient_checkpoint_chunk_size == 25
    engine._parameters = [SimpleNamespace(seq_len=sequence_length)]
    assert engine.gradient_checkpoint_chunk_size == min(25, sequence_length)
    engine.gradient_checkpoint_chunk_size = 17
    assert engine.gradient_checkpoint_chunk_size == 17


def test_short_sequence_default_prepares_and_differentiates_checkpoint_runner():
    inputs = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32)
    initial_state = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    cell = tf.keras.layers.SimpleRNNCell(
        1, dtype="float32", use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(0.7),
        recurrent_initializer=tf.keras.initializers.Constant(0.4),
    )
    sequence, final_state = tf.keras.layers.RNN(
        cell, return_sequences=True, return_state=True
    )(inputs, initial_state=[initial_state])
    model = tf.keras.Model((inputs, initial_state), (sequence, sequence, final_state))
    rnn = SimpleNamespace(seq_len=100, extractor_model=model, cell=SimpleNamespace())
    engine = TrainingEngine(rnn, 1, 1, training_approach="single")
    engine._parameters = [SimpleNamespace(seq_len=5)]
    engine.prepare_gradient_checkpointing()
    assert engine.gradient_checkpoint_chunk_size == 5
    values = tf.constant(np.arange(5, dtype=np.float32).reshape(1, 5, 1) / 10)
    state = tf.zeros((1, 1), tf.float32)

    def evaluate(checkpointed):
        with tf.GradientTape() as tape:
            outputs = (
                engine._extractor_forward(values, (state,))
                if checkpointed else model((values, state))
            )
            loss = tf.add_n([tf.reduce_sum(value) for value in outputs])
        return outputs, tape.gradient(loss, model.trainable_variables)

    expected, expected_gradients = evaluate(False)
    actual, actual_gradients = evaluate(True)
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)
    for left, right in zip(expected_gradients, actual_gradients):
        assert right is not None
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("cell_model", ["GLIF3Cell", "GLIF3", "default"])
@pytest.mark.parametrize("explicit", [None, False, True])
def test_migration_pins_previous_checkpoint_default(cell_model, explicit):
    config = {
        "rnn_cell_params": {"cell_model": cell_model},
        "training": {"n_epochs": 1},
    }
    if explicit is not None:
        config["training"]["gradient_checkpointing"] = explicit
    original = copy.deepcopy(config)
    migrated = preserve_previous_defaults(config)
    assert config == original
    assert migrated["training"]["gradient_checkpointing"] is (
        False if explicit is None else explicit
    )
    assert migrated["rnn_cell_params"]["dynamics_mode"] == "legacy"


def test_checked_in_legacy_training_examples_pin_checkpointing_off():
    examples = Path(__file__).resolve().parents[3] / "examples"
    paths = sorted((examples / "dpointnet_all2all").glob("config.train*.json"))
    paths.append(examples / "dpointnet_learning_rules/config.base.json")
    assert len(paths) == 5
    for path in paths:
        config = json.loads(path.read_text())
        assert config["rnn_cell_params"]["dynamics_mode"] == "legacy"
        assert config["training"]["gradient_checkpointing"] is False


def test_direct_cell_omitted_reset_matches_rnn_default():
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    args.pop("hard_reset", None)
    args.pop("hard_reset_gradient_mode", None)
    args["acceleration_profile"] = None
    cell = GLIF3Cell(network, inputs, **args)
    assert cell._hard_reset is True
    assert cell.hard_reset_gradient_mode == "soft_surrogate"


def test_renamed_aliases_reject_ambiguous_configurations():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="Specify only"):
            resolve_renamed_option("new", 1.0, "old", 1.0, 0.5)
    assert not caught


@pytest.mark.parametrize("value,expected", [(UNSET, 0.5), (0.0, 0.0), (1.0, 1.0)])
def test_standard_or_omitted_option_does_not_warn(value, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_renamed_option("new", value, "old", UNSET, 0.5) == expected
    assert not caught


@pytest.mark.parametrize("value,transform,expected", [
    (0.0, None, 0.0),
    (0.25, lambda value: 1 - np.clip(value, 0, 1), 0.75),
    (False, lambda value: "gaussian" if value else "triangular", "triangular"),
])
def test_legacy_option_warning_keeps_value_conversion(value, transform, expected):
    with pytest.warns(FutureWarning, match="old is deprecated; use new instead") as caught:
        actual = resolve_renamed_option("new", UNSET, "old", value, 0.5, transform)
    assert actual == expected
    assert caught[0].filename == __file__
    assert "No removal version is scheduled" in str(caught[0].message)


ALIASES = (
    ("gauss_std", "spike_surrogate_width", 0.4, 0.4),
    ("dampening_factor", "spike_surrogate_gain", 0.2, 0.2),
    ("recurrent_dampening_factor", "recurrent_spike_gradient_scale", 0.3, 0.3),
    ("voltage_gradient_dampening", "voltage_state_gradient_scale", 0.25, 0.75),
    ("pseudo_gauss", "spike_surrogate", False, "triangular"),
)


@pytest.mark.parametrize("legacy_name,name,value,expected", ALIASES)
def test_direct_cell_alias_warning_points_to_caller(legacy_name, name, value, expected):
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    for old, _, _, _ in ALIASES:
        args.pop(old, None)
    args[legacy_name] = value
    with pytest.warns(FutureWarning, match=f"{legacy_name} is deprecated; use {name}") as caught:
        cell = GLIF3Cell(network, inputs, **args)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert getattr(cell, name) == expected


def test_direct_cell_standard_names_are_silent():
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    for old, name, _, expected in ALIASES:
        args.pop(old, None)
        args[name] = expected
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        cell = GLIF3Cell(network, inputs, **args)
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
    for _, name, _, expected in ALIASES:
        assert getattr(cell, name) == expected


def test_legacy_positional_arguments_keep_their_slots_and_warn():
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    for old, _, _, _ in ALIASES:
        args.pop(old, None)
    with pytest.warns(FutureWarning) as caught:
        cell = GLIF3Cell(network, inputs, 1.0, 0.4, 0.2, 0.3, 0.25, 1.0, 1.0, 5, False, **args)
    assert len(caught) == 5
    for _, name, _, expected in ALIASES:
        assert getattr(cell, name) == expected


@pytest.mark.parametrize("voltage", [0.0, 0.25, 0.5, 1.0, -0.2, 1.2])
@pytest.mark.parametrize("surrogate", ["gaussian", "triangular"])
@pytest.mark.parametrize("fused", [False, True])
def test_legacy_names_preserve_forward_and_input_state_vjp(voltage, surrogate, fused):
    if fused and not (fused_cuda_available() and fused_nest_state_available()):
        pytest.skip("Matching native CUDA operators are required")
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    args.update(
        acceleration_profile=None, voltage_gradient_dampening=voltage,
        pseudo_gauss=surrogate == "gaussian",
        use_fused_cuda=fused, use_fused_state=fused,
        use_fused_nest_event_vjp=fused,
    )
    legacy = GLIF3Cell(network, inputs, **args)
    renamed = dict(args)
    for old, new in (
        ("gauss_std", "spike_surrogate_width"),
        ("dampening_factor", "spike_surrogate_gain"),
        ("recurrent_dampening_factor", "recurrent_spike_gradient_scale"),
    ):
        renamed[new] = renamed.pop(old)
    renamed["voltage_state_gradient_scale"] = 1.0 - np.clip(
        renamed.pop("voltage_gradient_dampening"), 0.0, 1.0
    )
    renamed["spike_surrogate"] = surrogate
    renamed.pop("pseudo_gauss")
    standard = GLIF3Cell(copy.deepcopy(network), copy.deepcopy(inputs), **renamed)
    assert legacy.voltage_state_gradient_scale == renamed["voltage_state_gradient_scale"]
    stimulus = tf.constant([[0.4, 0.8], [0.6, 0.2]], tf.float16)
    initial = tuple(legacy.zero_state(2, tf.float16))

    def evaluate(cell):
        differentiable = [value for value in initial if value.dtype.is_floating]
        with tf.GradientTape() as tape:
            tape.watch([stimulus, *differentiable])
            output, state = cell(stimulus, initial)
            loss = tf.add_n([
                tf.reduce_sum(tf.cast(value, tf.float32))
                for value in tf.nest.flatten((output, state))
                if value.dtype.is_floating
            ])
        return tf.nest.flatten((output, state)), tape.gradient(
            loss, [stimulus, *differentiable],
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

    old_values, old_gradients = evaluate(legacy)
    new_values, new_gradients = evaluate(standard)
    for old, new in zip(old_values + old_gradients, new_values + new_gradients):
        np.testing.assert_array_equal(old.numpy(), new.numpy())


def test_cell_omitted_scientific_defaults():
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    for name in (
        "dynamics_mode", "gauss_std", "dampening_factor",
        "recurrent_dampening_factor", "voltage_gradient_dampening",
        "pseudo_gauss", "detach_asc_reset", "train_recurrent_per_type",
    ):
        args.pop(name, None)
    args["acceleration_profile"] = None
    cell = GLIF3Cell(network, inputs, **args)
    assert cell.dynamics_mode == "nest"
    assert cell.spike_surrogate == "gaussian"
    assert cell.spike_surrogate_width == 0.28
    assert cell.spike_surrogate_gain == 0.05
    assert cell.recurrent_spike_gradient_scale == 1.0
    assert cell.voltage_state_gradient_scale == 1.0
    assert cell.state_precision == "selective"


@pytest.mark.parametrize("cell_model", ["GLIF3Cell", "GLIF3", "default"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_configuration_migration_preserves_previous_science(cell_model, dtype):
    original = {
        "run": {"dtype": dtype, "batch_size": 2, "seq_len": 7},
        "rnn_cell_params": {"cell_model": cell_model},
        "training": {
            "learning_rate": 0.004,
            "gradient_checkpointing": False,
            "parameters": [{"loss_functions": {"cost": 0.25}}],
        },
    }
    before = copy.deepcopy(original)
    migrated = preserve_previous_defaults(original)
    assert original == before
    assert migrated["run"] == before["run"]
    assert migrated["training"] == before["training"]
    cell = migrated["rnn_cell_params"]
    assert cell["dynamics_mode"] == "legacy"
    assert cell["spike_surrogate"] == "triangular"
    assert cell["spike_surrogate_gain"] == 0.3
    assert cell["recurrent_spike_gradient_scale"] == 0.5
    assert cell["voltage_state_gradient_scale"] == 0.5
    assert cell["detach_asc_reset"] is True
    assert cell["return_voltage_sequences"] is True
    assert cell["track_voltage_penalty"] is False
    assert cell["acceleration_profile"] == "auto"
    if dtype == "float16":
        assert cell["state_precision"] == "compute"


def test_migration_preserves_explicit_nest_inference_reset_and_voltage_convention():
    migrated = preserve_previous_defaults({
        "rnn_cell_params": {
            "dynamics_mode": "nest", "pseudo_gauss": True,
            "voltage_gradient_dampening": 0.25,
        },
    })
    assert migrated["run"]["dtype"] == "float32"
    cell = migrated["rnn_cell_params"]
    assert cell["hard_reset"] is True
    assert cell["spike_surrogate"] == "gaussian"
    assert cell["spike_surrogate_width"] == 0.5
    assert cell["voltage_state_gradient_scale"] == 0.75
    assert "voltage_gradient_dampening" not in cell


@pytest.mark.parametrize("supplied", [{"batch_size": 2}, {"seq_len": 5}, {}])
def test_project_batch_and_window_remain_required(supplied):
    rnn = RNN(**supplied)
    assert rnn.batch_size == supplied.get("batch_size")
    assert rnn.seq_len == supplied.get("seq_len")


@pytest.mark.skipif(not fused_cuda_available(), reason="Matching CUDA operators are required")
@pytest.mark.parametrize("explicit", [False, True])
def test_empty_recurrence_disables_only_automatically_selected_accumulation(explicit):
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    synapses = network["synapses"]
    for name in ("indices", "weights", "delays", "syn_ids"):
        synapses[name] = synapses[name][:0]
    args.update(acceleration_profile="auto", use_fused_state=True, _canonical_gradient_boundary=True)
    if explicit:
        args["use_fused_recurrent_accumulation"] = True
        with pytest.raises(ValueError, match="nonempty.*compact-pair"):
            GLIF3Cell(network, inputs, **args)
    else:
        cell = GLIF3Cell(network, inputs, **args)
        try:
            assert cell.use_fused_recurrent_accumulation is False
            assert cell.use_javier_recurrent_vjp is False
            assert cell.acceleration_report["selected"]["use_fused_recurrent_accumulation"] is False
            assert "nonempty" in cell.acceleration_report["reasons"]["use_fused_recurrent_accumulation"]
        finally:
            cell.close_fused_cuda()


@pytest.mark.skipif(not fused_cuda_available(), reason="Matching CUDA operators are required")
@pytest.mark.parametrize("selective", [False, True])
def test_direct_cell_auto_preserves_canonical_master_gradients(selective):
    network, inputs, args = make_cell(mode="nest", selective=selective, return_spec=True)
    for name in (
        "gauss_std", "dampening_factor", "recurrent_dampening_factor",
        "voltage_gradient_dampening", "pseudo_gauss", "use_fused_state",
        "acceleration_profile", "detach_reset", "state_precision",
    ):
        args.pop(name)
    reference = GLIF3Cell(copy.deepcopy(network), copy.deepcopy(inputs), acceleration_profile=None, **args)
    automatic = GLIF3Cell(copy.deepcopy(network), copy.deepcopy(inputs), **args)
    try:
        dtype = tf.as_dtype(reference.compute_dtype)
        initial = list(reference.zero_state(2, dtype))
        initial[0] = tf.concat([
            tf.ones((2, 1), dtype),
            tf.zeros((2, initial[0].shape[1] - 1), dtype),
        ], axis=1)
        stimulus = tf.constant([[0.4, 0.8], [0.6, 0.2]], dtype)

        def evaluate(cell):
            floating = [value for value in initial if value.dtype.is_floating]
            masters = [cell.recurrent_weight_values, cell.inputs["drive"]["input_weight_values"]]
            with tf.GradientTape() as tape:
                tape.watch([stimulus, *floating])
                output = cell(stimulus, initial)
                loss = tf.add_n([
                    tf.reduce_sum(tf.cast(value, tf.float32))
                    for value in tf.nest.flatten(output) if value.dtype.is_floating
                ])
            gradients = tape.gradient(
                loss, [stimulus, *floating, *masters],
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            return tf.nest.flatten(output) + gradients

        for actual, expected in zip(evaluate(automatic), evaluate(reference)):
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=3e-4)
        assert automatic._use_direct_csr_recurrent_gradient is False
        assert automatic.use_fused_recurrent_accumulation is False
    finally:
        reference.close_fused_cuda()
        automatic.close_fused_cuda()


def test_inference_rnn_auto_requires_a_training_gradient_boundary(monkeypatch):
    from types import SimpleNamespace
    from bmtk.simulator.dpointnet import acceleration, alpha_basis

    network, inputs, args = make_cell(mode="nest", return_spec=True)
    args.pop("acceleration_profile")
    rnn = RNN(batch_size=2, seq_len=3, cell_params=args)
    monkeypatch.setattr(RNN, "recurrent_network", property(lambda self: network))
    monkeypatch.setattr(alpha_basis, "prepare_alpha_basis", lambda *args: None)
    resolve = acceleration.resolve_acceleration_options

    def check_boundary(*args, **kwargs):
        assert kwargs["canonical_gradient_boundary"] is False
        return resolve(*args, **kwargs)

    monkeypatch.setattr(acceleration, "resolve_acceleration_options", check_boundary)
    rnn._input_networks = {
        "drive": SimpleNamespace(name="drive", n_spiking_nodes=2, to_dict=lambda: inputs["drive"])
    }
    try:
        rnn.build()
        assert rnn.cell._use_direct_csr_recurrent_gradient is False
        assert rnn.cell.use_fused_recurrent_accumulation is False
    finally:
        if rnn.cell is not None:
            rnn.cell.close_fused_cuda()


@pytest.mark.parametrize("keras_master", [False, True])
def test_gradient_restore_recognizes_distributed_master_components(monkeypatch, keras_master):
    from types import SimpleNamespace
    from bmtk.simulator.dpointnet.cell_models import glif3_cell

    strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])
    with strategy.scope():
        if keras_master:
            master = tf.keras.layers.Layer(dtype="float32").add_weight(
                name="master", shape=(2,), initializer="ones", dtype="float32",
            )
            underlying = master.value if not callable(master.value) else master
        else:
            master = tf.Variable([1.0, 1.0], dtype=tf.float32)
            underlying = master
    local_master = strategy.experimental_local_results(underlying)[0]
    other = tf.Variable([0.0, 0.0], dtype=tf.float32)
    cell = SimpleNamespace(recurrent_weight_values=master, recurrent_fused_connectivity={})
    monkeypatch.setattr(glif3_cell, "restore_csr_values", lambda gradient, connectivity: tf.reverse(gradient, [0]))
    gradients = (tf.constant([10.0, 20.0]), tf.constant([3.0, 4.0]))
    actual = GLIF3Cell.restore_segmented_variable_gradients(
        cell, (local_master, other), gradients,
    )
    np.testing.assert_array_equal(actual[0], [20.0, 10.0])
    np.testing.assert_array_equal(actual[1], [3.0, 4.0])
    with pytest.raises(RuntimeError, match="could not locate its FP32 master"):
        GLIF3Cell.restore_segmented_variable_gradients(cell, (other,), (gradients[1],))
