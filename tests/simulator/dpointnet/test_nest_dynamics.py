import copy
import gc
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

tf = pytest.importorskip("tensorflow")
from bmtk.simulator.dpointnet.cell_models.nest_dynamics import (
    integration_coefficients,
    active_update,
    spike_reset,
    time_steps,
)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("refractory_dtype", [tf.int8, tf.int16])
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize(
    "batch,dampening,loss_outputs",
    [(3, 0.25, None), (32, -0.2, (0, 6)), (1, 1.2, (1,))],
)
def test_fused_nest_state_matches_reference_values_and_gradients(
    dtype, refractory_dtype, hard_reset, batch, dampening, loss_outputs
):
    from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import (
        fused_nest_state,
        fused_nest_state_available,
    )
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
        spike_function,
        straight_through_dampen,
    )

    if not fused_nest_state_available():
        pytest.skip("Fused NEST state operator is unavailable")
    rng = np.random.default_rng(813)
    neurons = 5
    long_refractory = 300 if refractory_dtype == tf.int16 else 4

    def constant(shape, low, high):
        return tf.constant(rng.uniform(low, high, shape), dtype)

    parameters = dict(
        syn_decay=constant((neurons, 4), 0.7, 0.9),
        psc_initial=constant((neurons, 4), 0.2, 0.4),
        asc_decay=constant((neurons, 2), 0.7, 0.9),
        asc_amps=constant((neurons, 2), -0.2, 0.0),
        decay=constant((neurons,), 0.8, 0.95),
        current_factor=constant((neurons,), 0.05, 0.1),
        asc_mean=constant((neurons, 2), 0.8, 0.95),
        asc_refractory_decay=constant((neurons, 2), 0.5, 0.8),
        psc_voltage=constant((neurons, 4), 0.02, 0.06),
        rise_voltage=constant((neurons, 4), 0.01, 0.03),
        t_ref_steps=tf.constant([2, 3, long_refractory, 2, 3], refractory_dtype),
        dt=tf.cast(0.5, dtype),
        v_reset=constant((neurons,), -0.2, 0.2),
        v_th=tf.cast(1.0, dtype),
        dampening=tf.cast(0.3, dtype),
        voltage_gradient_dampening=tf.cast(dampening, dtype),
        hard_reset=hard_reset,
    )
    refractory = tf.constant([[0, long_refractory, 0, 2, 0]] * batch, refractory_dtype)
    values = [
        tf.constant([[0.4, 1.4, 1.4, 0.8, 1.0]] * batch, dtype),
        constant((batch, neurons * 2), -0.1, 0.1),
        constant((batch, neurons * 4), -0.2, 0.3),
        constant((batch, neurons * 4), -0.2, 0.3),
        constant((batch, neurons * 4), -0.2, 0.3),
        tf.cast(rng.integers(0, 2, (batch, neurons * 3)), dtype),
    ]

    def reference(voltage, adaptation, rise, psc, currents, history):
        new_rise = (
            tf.reshape(rise, (batch, neurons, 4)) * parameters["syn_decay"]
            + tf.reshape(currents, (batch, neurons, 4)) * parameters["psc_initial"]
        )
        new_psc = tf.reshape(psc, (batch, neurons, 4)) * parameters[
            "syn_decay"
        ] + parameters["dt"] * parameters["syn_decay"] * tf.reshape(
            rise, (batch, neurons, 4)
        )
        voltage, remaining, adaptation, active = active_update(
            straight_through_dampen(voltage, parameters["voltage_gradient_dampening"]),
            refractory,
            tf.reshape(adaptation, (batch, neurons, 2)),
            tf.reshape(psc, (batch, neurons, 4)),
            tf.reshape(rise, (batch, neurons, 4)),
            **{
                key: parameters[key]
                for key in (
                    "decay",
                    "current_factor",
                    "asc_decay",
                    "asc_mean",
                    "psc_voltage",
                    "rise_voltage",
                    "hard_reset",
                )
            },
            reset_voltage=parameters["v_reset"],
        )
        spikes = spike_function(voltage - parameters["v_th"], parameters["dampening"])
        spikes = tf.where(active, spikes, tf.zeros_like(spikes))
        voltage, remaining, adaptation = spike_reset(
            voltage,
            remaining,
            adaptation,
            spikes,
            reset_voltage=parameters["v_reset"],
            refractory_steps=parameters["t_ref_steps"],
            asc_amplitudes=parameters["asc_amps"],
            asc_refractory_decay=parameters["asc_refractory_decay"],
            hard_reset=hard_reset,
        )
        return (
            spikes,
            voltage,
            remaining,
            tf.reshape(adaptation, (batch, neurons * 2)),
            tf.reshape(new_rise, (batch, neurons * 4)),
            tf.reshape(new_psc, (batch, neurons * 4)),
            tf.concat([spikes, history[:, :-neurons]], axis=1),
        )

    def evaluate(fused):
        with tf.GradientTape() as tape:
            tape.watch(values)
            if fused:
                outputs = fused_nest_state(
                    values[0], refractory, *values[1:], **parameters
                )
            else:
                outputs = reference(*values)
            loss = tf.add_n(
                [
                    tf.reduce_sum(tf.square(tf.cast(output, tf.float32)))
                    * (index + 1)
                    / 16
                    for index, output in enumerate(outputs)
                    if output.dtype.is_floating
                    and (loss_outputs is None or index in loss_outputs)
                ]
            )
        return outputs, tape.gradient(
            loss, values, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )

    with tf.device("/GPU:0"):
        expected, expected_gradients = tf.function(lambda: evaluate(False))()
        fused_function = tf.function(lambda: evaluate(True)).get_concrete_function()
        actual, gradients = fused_function()
    backward_ops = [
        operation
        for operation in fused_function.graph.get_operations()
        if operation.type == "DpointnetNestStateBackward"
    ]
    assert len(backward_ops) == 1
    backward_mask = backward_ops[0].inputs[1]
    assert backward_mask.op.type == "Cast"
    assert backward_mask.op.inputs[0].dtype == dtype
    assert backward_mask.dtype == refractory_dtype
    tolerance = 3e-3 if dtype == tf.float16 else 1e-6
    for observed, reference_value in zip(actual, expected):
        assert observed.dtype == reference_value.dtype
        np.testing.assert_allclose(
            observed, reference_value, rtol=tolerance, atol=tolerance
        )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[2], expected[2])
    np.testing.assert_array_equal(actual[-1], expected[-1])
    assert np.any(expected[0].numpy() != 0)
    for observed, reference_value in zip(gradients, expected_gradients):
        assert observed is not None and reference_value is not None
        np.testing.assert_allclose(
            observed, reference_value, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("input_index", range(10))
def test_nest_state_op_rejects_malformed_shapes(backward, input_index):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if not glif_state_ops.fused_nest_state_available():
        pytest.skip("Fused NEST state operator is unavailable")
    shapes = (
        [(2, 3), (2, 3), (3, 28), (), (), (2, 3), (2, 3), (2, 6), (2, 12), (2, 12)]
        if backward
        else [(2, 3), (2, 3), (2, 6), (2, 12), (2, 12), (2, 12), (3, 28), (3,), (), ()]
    )
    arguments = [
        tf.zeros(
            shape,
            tf.int16 if index == 1 or (not backward and index == 7) else tf.float32,
        )
        for index, shape in enumerate(shapes)
    ]
    arguments[input_index] = tf.zeros((2,), arguments[input_index].dtype)
    operator = (
        glif_state_ops._OPS.dpointnet_nest_state_backward
        if backward
        else glif_state_ops._OPS.dpointnet_nest_state_forward
    )
    with tf.device("/GPU:0"), pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        operator(*arguments)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("history_shape", [(2, 0), (1, 4), (2, 3), (8,)])
def test_nest_spike_history_rejects_malformed_shapes(backward, history_shape):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if not glif_state_ops.fused_nest_state_available():
        pytest.skip("Fused NEST state operator is unavailable")
    with tf.device("/GPU:0"), pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError)
    ):
        voltage = tf.zeros((2, 2))
        refractory = tf.zeros((2, 2), tf.bool)
        history = tf.zeros(history_shape)
        if backward:
            glif_state_ops._OPS.dpointnet_spike_shift_backward(
                voltage, refractory, voltage, history, tf.constant(0.3)
            )
        else:
            glif_state_ops._OPS.dpointnet_spike_shift(voltage, refractory, history)


def test_nest_state_availability_rejects_stale_library(monkeypatch):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    monkeypatch.setattr(glif_state_ops, "fused_glif_state_available", lambda: True)
    monkeypatch.setattr(glif_state_ops, "_OPS", SimpleNamespace())
    assert not glif_state_ops.fused_nest_state_available()
    with pytest.raises(RuntimeError, match="rebuilt CUDA operators"):
        glif_state_ops.fused_nest_state(
            *([None] * 7),
            **dict.fromkeys(
                [
                    "syn_decay",
                    "psc_initial",
                    "asc_decay",
                    "asc_amps",
                    "decay",
                    "current_factor",
                    "asc_mean",
                    "asc_refractory_decay",
                    "psc_voltage",
                    "rise_voltage",
                    "t_ref_steps",
                    "dt",
                    "v_reset",
                    "v_th",
                    "dampening",
                    "voltage_gradient_dampening",
                ]
            ),
        )


def make_network_inputs(delay=1.0, internal_noise=False):
    nodes = {
        "C_m": np.array([100.0]),
        "g": np.array([10.0]),
        "E_L": np.array([-70.0]),
        "V_m": np.array([-70.0]),
        "V_th": np.array([-50.0]),
        "V_reset": np.array([-70.0]),
        "t_ref": np.array([2.0]),
        "k": np.array([[0.01, 0.1]]),
        "asc_amps": np.array([[0.0, 0.0]]),
    }
    network = {
        "n_nodes": 1,
        "node_type_ids": np.array([0]),
        "node_params": nodes,
        "synapses": {
            "indices": np.array([[0, 0]], dtype=np.uint32),
            "weights": np.array([0.0]),
            "delays": np.array([2.0]),
            "dense_shape": (1, 1),
            "syn_ids": np.array([0]),
            "dynamics_params": {"basis_weights": [[1.0]]},
        },
    }
    inputs = {
        "drive": {
            "n_inputs": 1,
            "indices": np.array([[0, 0]]),
            "weights": np.array([10.0]),
            "delays": np.array([delay]),
            "syn_ids": np.array([0]),
            "input_type": "spikes",
            "options": {"trainable": True},
        }
    }
    if internal_noise:
        inputs["drive"]["input_type"] = "poisson_spikes_internal"
        inputs["drive"]["options"]["firing_rate"] = 250.0
    return network, inputs


def make_cell(
    dt=1.0,
    delay=1.0,
    hard_reset=True,
    mode="nest",
    internal_noise=False,
    use_fused_cuda=False,
):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    network, inputs = make_network_inputs(delay, internal_noise)
    cell_options = {}
    if mode is not None:
        cell_options["dynamics_mode"] = mode
    return GLIF3Cell(
        network,
        inputs,
        dt=dt,
        tau_basis=[2.0],
        hard_reset=hard_reset,
        train_recurrent_per_type=False,
        use_fused_cuda=use_fused_cuda,
        **cell_options,
    )


@pytest.mark.parametrize(
    "build_mode", ["configured_training", "explicit_training", "inference"]
)
@pytest.mark.parametrize("mode", ["nest", "legacy"])
def test_rnn_build_resolves_reset_for_training_and_inference(build_mode, mode):
    from bmtk.simulator.dpointnet.rnn_model import RNN

    network, inputs = make_network_inputs()
    rnn = RNN(
        seq_len=4,
        batch_size=1,
        cell_params={"dynamics_mode": mode, "tau_basis": [2.0]},
    )
    rnn._recurrent_networks["test"] = SimpleNamespace(
        to_dict=lambda: copy.deepcopy(network)
    )
    rnn._input_networks["drive"] = SimpleNamespace(
        name="drive", n_spiking_nodes=1, to_dict=lambda: copy.deepcopy(inputs["drive"])
    )
    if build_mode == "configured_training":
        engine = rnn.set_training(rnn=rnn, n_epochs=1, steps_per_epoch=1)
        engine.add_parameters("test", batch_size=1, seq_len=4)
    try:
        if build_mode == "explicit_training":
            rnn.build(training=True)
        else:
            rnn.build()
        expected_hard_reset = mode == "nest" and build_mode == "inference"
        assert rnn.cell._hard_reset is expected_hard_reset
        assert "hard_reset" not in rnn.cell_params
        if expected_hard_reset:
            with pytest.raises(ValueError, match="already built with hard reset"):
                rnn._prepare_training_model()
        else:
            rnn._prepare_training_model()
            assert rnn.cell._hard_reset is False
    finally:
        rnn.cleanup()


def test_default_dynamics_mode_preserves_legacy_behavior():
    cell = make_cell(mode=None, hard_reset=None)

    assert cell.dynamics_mode == "legacy"
    assert cell._hard_reset is False
    assert len(cell.zero_state(1, tf.float32)) == 7


def test_full_cell_delays_and_input_gradient():
    immediate = make_cell(delay=1.0)
    delayed = make_cell(delay=3.0)

    def run(cell):
        state = cell.zero_state(1, tf.float32)
        voltages = []
        for step in range(8):
            output, state = cell(tf.constant([[1.0 if step == 0 else 0.0]]), state)
            voltages.append(output[0, 1])
        return tf.stack(voltages), state

    with tf.GradientTape() as tape:
        values, state = run(delayed)
        loss = tf.reduce_sum(values)
    gradient = tape.gradient(loss, delayed.inputs["drive"]["input_weight_values"])
    baseline, _ = run(immediate)
    np.testing.assert_allclose(values[2:], baseline[:-2], atol=1e-8)
    assert len(state) == 8 and state[6].numpy()[0] == 8
    assert np.isfinite(gradient.numpy()).all() and np.any(gradient.numpy() != 0)


def test_submillisecond_recurrent_delay_buffer_has_step_units():
    cell = make_cell(dt=0.25, delay=0.5)
    assert cell.max_delay == 8
    assert tuple(cell.recurrent_dense_shape) == (1, 8)
    assert cell.recurrent_indices.numpy()[0, 1] == 7


def test_refractory_quantization_uses_nest_ticks_then_ceiling():
    np.testing.assert_array_equal(
        time_steps([1.65, 3.25, 3.2500000000000004], 1.0), [2, 4, 4]
    )
    np.testing.assert_array_equal(
        time_steps([1.65, 3.25, 3.2500000000000004], 0.25), [7, 13, 13]
    )


def test_cell_rnn_chunk_continuation_and_soft_reset_training():
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    cell = make_cell(delay=3.0, hard_reset=False)
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    layer._autocast = False
    values = tf.constant([[[2.0], [0.0], [1.0], [0.0], [0.0], [3.0], [0.0], [0.0]]])
    initial = cell.zero_state(1, tf.float32)
    with tf.GradientTape() as tape:
        full = layer(values, initial_state=initial)
        loss = tf.reduce_sum(full[0][..., 1])
    gradient = tape.gradient(loss, cell.inputs["drive"]["input_weight_values"])
    assert (
        gradient is not None and np.isfinite(gradient).all() and np.any(gradient != 0)
    )
    first = layer(values[:, :4], initial_state=initial)
    second = layer(values[:, 4:], initial_state=first[1:])
    np.testing.assert_allclose(
        full[0], tf.concat([first[0], second[0]], axis=1), atol=1e-7
    )
    for expected, actual in zip(full[1:], second[1:]):
        np.testing.assert_allclose(expected, actual, atol=1e-7)


@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("return_state", [False, True])
@pytest.mark.parametrize("unroll,explicit_state", [(False, True), (True, False)])
def test_explicit_rnn_matches_direct_cell_values_and_gradients(
    return_sequences, return_state, unroll, explicit_state
):
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    cell = make_cell(delay=3.0, hard_reset=False)
    layer = ExplicitStateRNN(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        unroll=unroll,
    )
    layer._autocast = False
    values = tf.constant([[[2.0], [0.0], [1.0], [0.0], [0.0], [3.0], [0.0], [0.0]]])
    initial = cell.zero_state(1, tf.float32)
    variables = [
        values,
        cell.recurrent_weight_values,
        cell.inputs["drive"]["input_weight_values"],
    ]
    with tf.GradientTape() as tape:
        tape.watch(values)
        actual = layer(values, initial_state=initial if explicit_state else None)
        actual_output = actual[0] if return_state else actual
        loss = tf.reduce_sum(actual_output[..., 1])
    actual_gradients = tape.gradient(loss, variables)

    with tf.GradientTape() as tape:
        tape.watch(values)
        reference_state = initial
        reference_outputs = []
        for inputs in tf.unstack(values, axis=1):
            output, reference_state = cell(inputs, reference_state)
            reference_outputs.append(output)
        reference_output = (
            tf.stack(reference_outputs, axis=1)
            if return_sequences
            else reference_outputs[-1]
        )
        reference_loss = tf.reduce_sum(reference_output[..., 1])
    reference_gradients = tape.gradient(reference_loss, variables)

    np.testing.assert_allclose(actual_output, reference_output, rtol=1e-5, atol=1e-7)
    if return_state:
        for actual_state, expected_state in zip(actual[1:], reference_state):
            assert actual_state.dtype == expected_state.dtype
            np.testing.assert_allclose(
                actual_state, expected_state, rtol=1e-5, atol=1e-7
            )
    assert actual_gradients[0] is None and reference_gradients[0] is None
    for actual_gradient, expected_gradient in zip(
        actual_gradients[1:], reference_gradients[1:]
    ):
        assert actual_gradient is not None and expected_gradient is not None
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=1e-5, atol=1e-7
        )
    assert np.any(actual_gradients[-1].numpy() != 0)


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("return_sequences", [False, True])
@pytest.mark.parametrize("return_state", [False, True])
@pytest.mark.parametrize("unroll", [False, True])
def test_nest_compact_outputs_preserve_mixed_precision(
    policy, return_sequences, return_state, unroll
):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
        voltage_penalty_mean_step,
    )
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.rnn_model import RNN

    old_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy(policy)
        cell = make_cell(delay=3.0, hard_reset=False)
        values = tf.constant(
            [[[400.0], [0.0], [200.0], [0.0], [0.0], [600.0], [0.0], [0.0]]],
            cell.compute_dtype,
        )
        initial = cell.zero_state(1, cell.compute_dtype)
        targets = [
            cell.recurrent_weight_values,
            cell.inputs["drive"]["input_weight_values"],
        ]
        with tf.GradientTape() as tape:
            state = initial
            packed = []
            for inputs in tf.unstack(values, axis=1):
                output, state = cell(inputs, state)
                penalty = voltage_penalty_mean_step(output[..., 1:], 1)
                packed.append(
                    tf.concat(
                        [tf.cast(output[..., :1], tf.float32), penalty[:, None]],
                        axis=-1,
                    )
                )
            reference = tf.stack(packed, axis=1) if return_sequences else packed[-1]
            reference_loss = 0.7 * tf.reduce_sum(
                reference[..., :1]
            ) + 1.3 * tf.reduce_sum(reference[..., 1])
        expected_gradients = tape.gradient(reference_loss, targets)

        cell._track_voltage_penalty = True
        cell._return_voltage_sequences = False
        layer = ExplicitStateRNN(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            unroll=unroll,
        )
        layer._autocast = False
        symbolic = tf.keras.Input(shape=(8, 1), dtype=values.dtype)
        layer_outputs = layer(symbolic)
        model = tf.keras.Model(symbolic, layer_outputs)
        rnn = RNN()
        rnn._cell = cell
        split_outputs, split_states = rnn._split_rnn_layer_output(layer_outputs)
        assert len(split_outputs) == 2
        assert len(split_states) == (len(initial) if return_state else 0)
        extracted_outputs, extracted_states = rnn._split_rnn_layer_output(layer.output)
        assert len(extracted_outputs) == 2
        assert len(extracted_states) == len(split_states)
        for actual_spec, expected_spec in zip(
            extracted_outputs + extracted_states, split_outputs + split_states
        ):
            assert actual_spec.shape == expected_spec.shape
            assert actual_spec.dtype == expected_spec.dtype
        with tf.GradientTape() as tape:
            actual = model(values)
            outputs = actual[0] if return_state else actual
            assert isinstance(outputs, (tuple, list)) and len(outputs) == 2
            spikes, penalty = outputs
            assert spikes.dtype == values.dtype and penalty.dtype == tf.float32
            loss = 0.7 * tf.reduce_sum(
                tf.cast(spikes, tf.float32)
            ) + 1.3 * tf.reduce_sum(penalty)
        gradients = tape.gradient(loss, targets)
        np.testing.assert_array_equal(spikes, reference[..., :1])
        np.testing.assert_array_equal(penalty, reference[..., 1])
        assert np.count_nonzero(tf.stack(packed)[..., 0]) > 0
        assert np.max(tf.stack(packed)[..., 1]) > 0
        for actual_gradient, expected_gradient in zip(gradients, expected_gradients):
            assert actual_gradient is not None and expected_gradient is not None
            np.testing.assert_allclose(
                actual_gradient, expected_gradient, rtol=1e-5, atol=1e-7
            )
        if return_state:
            for actual_state, expected_state in zip(actual[1:], state):
                assert actual_state.dtype == expected_state.dtype
                np.testing.assert_array_equal(actual_state, expected_state)
        assert model.outputs[0].dtype == values.dtype.name
        assert model.outputs[1].dtype == "float32"
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)


@pytest.mark.parametrize("available", [False, True])
def test_nest_fused_state_selection_and_unsupported_models(monkeypatch, available):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import _resolve_fused_state

    monkeypatch.setattr(
        "bmtk.simulator.dpointnet.cell_models.glif3_cell.fused_nest_state_available",
        lambda: available,
    )
    assert _resolve_fused_state(False, 4, False, "nest") is False
    assert _resolve_fused_state("auto", 4, False, "nest") is available
    if available:
        assert _resolve_fused_state(True, 4, False, "nest") is True
    else:
        with pytest.raises(ValueError, match="rebuild CUDA"):
            _resolve_fused_state(True, 4, False, "nest")
    for basis, gaussian in ((1, False), (4, True)):
        assert _resolve_fused_state("auto", basis, gaussian, "nest") is False
        with pytest.raises(ValueError, match="incompatible"):
            _resolve_fused_state(True, basis, gaussian, "nest")


@pytest.mark.parametrize("dt", [1.0, 0.25])
@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("hard_reset,batch_size", [(True, 1), (False, 5)])
@pytest.mark.parametrize("drive_scale", [0.05, 1.0])
def test_nest_cuda_matches_cpu_and_gpu_fallback(
    dt, policy, hard_reset, batch_size, drive_scale, record_property
):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available

    if not fused_cuda_available():
        pytest.skip("Fused CUDA operators unavailable")
    network = {
        "n_nodes": 3,
        "node_type_ids": np.array([0, 1, 0]),
        "node_params": {
            "C_m": np.array([100.0, 80.0]),
            "g": np.array([10.0, 8.0]),
            "E_L": np.array([-70.0, -65.0]),
            "V_m": np.array([-69.0, -64.0]),
            "V_th": np.array([-50.0, -45.0]),
            "V_reset": np.array([-68.0, -64.0]),
            "t_ref": np.array([2.0, 1.65]),
            "k": np.array([[0.01, 0.1], [0.02, 0.15]]),
            "asc_amps": np.array([[-10.0, -50.0], [-20.0, -40.0]]),
            "asc_init": np.array([[-1.0, -2.0], [-2.0, -1.0]]),
        },
        "synapses": {
            "indices": np.array([[0, 0], [0, 1], [1, 0], [2, 1], [2, 2]]),
            "weights": np.array([80.0, -100.0, 40.0, -60.0, 50.0]),
            "delays": np.array([1.0, 2.0, 3.0, 1.0, 2.0]),
            "dense_shape": (3, 3),
            "syn_ids": np.array([0, 1, 0, 1, 0]),
            "dynamics_params": {
                "basis_weights": [[1.0, 0.3, 0.1, 0.05], [0.1, 1.0, 0.2, 0.1]]
            },
        },
    }
    inputs = {
        "drive": {
            "n_inputs": 4,
            "indices": np.array(
                [[0, 0], [0, 1], [1, 1], [2, 0], [2, 2]], dtype=np.uint32
            ),
            "weights": np.array([1200.0, 800.0, 1000.0, 900.0, 700.0]),
            "delays": np.array([1.0, 3.0, 2.0, 1.0, 3.0]),
            "syn_ids": np.array([0, 1, 0, 1, 0]),
            "input_type": "spikes",
            "options": {"trainable": True},
        }
    }
    steps = int(40 / dt)
    inputs["drive"]["weights"] *= drive_scale
    values = np.zeros((batch_size, steps, 4), np.float32)
    values[:, (np.array([0.0, 5.0, 13.0, 25.0]) / dt).astype(int), :3] = [1.0, 2.0, 1.0]
    values *= np.arange(1, batch_size + 1, dtype=np.float32)[:, None, None]
    old_policy = tf.keras.mixed_precision.global_policy()
    results = {}
    tolerance = 2e-2 if policy == "mixed_float16" else 1e-5
    try:
        tf.keras.mixed_precision.set_global_policy(policy)
        for label, device, fused, fused_state in (
            ("cpu", "/CPU:0", False, False),
            ("gpu_fallback", "/GPU:0", False, False),
            ("cuda", "/GPU:0", True, False),
            ("cuda_state", "/GPU:0", True, True),
        ):
            started = time.perf_counter()
            with tf.device(device):
                cell = GLIF3Cell(
                    copy.deepcopy(network),
                    copy.deepcopy(inputs),
                    dt=dt,
                    tau_basis=[2.0, 6.0, 10.0, 20.0],
                    dynamics_mode="nest",
                    hard_reset=hard_reset,
                    use_fused_cuda=fused,
                    use_fused_state=fused_state,
                    train_recurrent_per_type=False,
                    batch_size=batch_size,
                )
                assert cell._use_fused_cuda is fused
                assert cell._use_fused_state is fused_state
                layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
                layer._autocast = False
                sequence = tf.constant(values, dtype=cell.compute_dtype)
                initial = list(cell.zero_state(batch_size, cell.compute_dtype))
                initial[6] = tf.fill((batch_size,), 4095)

                @tf.function(
                    input_signature=[
                        tf.TensorSpec((batch_size, None, 4), sequence.dtype),
                        [tf.TensorSpec(value.shape, value.dtype) for value in initial],
                    ]
                )
                def rollout(sequence, initial):
                    return layer(sequence, initial_state=initial)

                with tf.GradientTape() as tape:
                    full = rollout(sequence, initial)
                    loss = tf.reduce_mean(tf.cast(full[0][..., 3:], tf.float32))
                gradients = tape.gradient(
                    loss,
                    (
                        cell.recurrent_weight_values,
                        cell.inputs["drive"]["input_weight_values"],
                    ),
                )
                split = int(7 / dt)
                first = rollout(sequence[:, :split], initial)
                second = rollout(sequence[:, split:], list(first[1:]))
                np.testing.assert_allclose(
                    full[0],
                    tf.concat([first[0], second[0]], axis=1),
                    rtol=tolerance,
                    atol=tolerance,
                )
                for expected, actual in zip(full[1:], second[1:]):
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                for gradient in gradients:
                    assert gradient is not None
                    assert np.isfinite(gradient).all() and np.any(gradient != 0)
                results[label] = (
                    [value.numpy() for value in full],
                    [value.numpy() for value in gradients],
                )
                spike_count = np.count_nonzero(full[0][..., :3])
                record_property(f"{label}_spikes", int(spike_count))
                if drive_scale == 1.0:
                    assert spike_count > 0
                np.testing.assert_array_equal(
                    full[7], np.full(batch_size, 4095 + steps)
                )
            record_property(f"{label}_seconds", time.perf_counter() - started)
            cell.close_fused_cuda()
            del rollout, layer, cell
            gc.collect()
        for actual, expected in zip(results["cuda_state"][0], results["cuda"][0]):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        np.testing.assert_array_equal(
            results["cuda_state"][0][0][..., :3], results["cuda"][0][0][..., :3]
        )
        for actual, expected in zip(results["cuda_state"][1], results["cuda"][1]):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        expected_outputs, expected_gradients = results["cpu"]
        for label in ("gpu_fallback", "cuda"):
            outputs, gradients = results[label]
            voltage_delta = np.abs(
                outputs[0][..., 3:].astype(np.float64) - expected_outputs[0][..., 3:]
            )
            record_property(
                f"{label}_voltage_max_abs_mv", float(np.max(voltage_delta) * 20)
            )
            record_property(
                f"{label}_voltage_max_scaled_error",
                float(
                    np.max(
                        voltage_delta
                        / np.maximum(1, np.abs(expected_outputs[0][..., 3:]))
                    )
                ),
            )
            spike_differences = []
            for sample in range(batch_size):
                for neuron in range(3):
                    actual_steps = np.flatnonzero(outputs[0][sample, :, neuron])
                    expected_steps = np.flatnonzero(
                        expected_outputs[0][sample, :, neuron]
                    )
                    np.testing.assert_equal(actual_steps.size, expected_steps.size)
                    spike_differences.extend(np.abs(actual_steps - expected_steps))
            max_spike_error = max(spike_differences, default=0)
            assert max_spike_error <= 1
            spikes_match = max_spike_error == 0
            record_property(f"{label}_spike_times_exact", bool(spikes_match))
            record_property(
                f"{label}_max_spike_time_error_ms", float(max_spike_error * dt)
            )
            record_property(
                f"{label}_jittered_spikes", int(np.count_nonzero(spike_differences))
            )
            record_property(f"{label}_continuous_parity_checked", bool(spikes_match))
            for index, (actual, expected) in enumerate(zip(outputs, expected_outputs)):
                record_property(
                    f"{label}_state_{index}_max_abs",
                    float(np.max(np.abs(actual.astype(np.float64) - expected))),
                )
                if index in (7, 8) or (spikes_match and index in (1, 3)):
                    np.testing.assert_array_equal(actual, expected)
                elif spikes_match:
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
            for index, (actual, expected) in enumerate(
                zip(gradients, expected_gradients)
            ):
                record_property(
                    f"{label}_gradient_{index}_max_abs",
                    float(np.max(np.abs(actual - expected))),
                )
                if spikes_match:
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
def test_nest_fused_state_checkpointed_poisson_replay(policy):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.custom_ops import fused_nest_state_available
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner

    if not fused_nest_state_available():
        pytest.skip("Fused NEST state operator is unavailable")
    old_policy = tf.keras.mixed_precision.global_policy()
    results = []
    try:
        tf.keras.mixed_precision.set_global_policy(policy)
        for fused in (False, True):
            with tf.device("/GPU:0"):
                network, inputs = make_network_inputs(delay=3.0, internal_noise=True)
                network["synapses"]["dynamics_params"]["basis_weights"] = [
                    [1, 0.3, 0.1, 0.05]
                ]
                network["synapses"]["weights"][:] = 40
                inputs["drive"]["weights"][:] = 1200
                inputs["drive"]["options"]["firing_rate"] = 1000
                cell = GLIF3Cell(
                    network,
                    inputs,
                    dt=0.25,
                    tau_basis=[2, 6, 10, 20],
                    dynamics_mode="nest",
                    hard_reset=False,
                    batch_size=32,
                    train_recurrent_per_type=False,
                    use_fused_cuda=True,
                    use_fused_state=fused,
                    return_voltage_sequences=False,
                    track_voltage_penalty=True,
                )
                initial = list(cell.zero_state(32, cell.compute_dtype))
                initial[6] = tf.fill((32,), 4095)
                sequence = tf.zeros((32, 31, 0), cell.compute_dtype)
                sequence_input = tf.keras.Input(shape=(None, 0), dtype=sequence.dtype)
                state_inputs = [
                    tf.keras.Input(shape=value.shape[1:], dtype=value.dtype)
                    for value in initial
                ]
                layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
                layer._autocast = False
                layer_outputs = layer(sequence_input, initial_state=state_inputs)
                core = tf.keras.Model(
                    [sequence_input, *state_inputs],
                    [*layer_outputs[0], *layer_outputs[1:]],
                )
                runner = SegmentedRecomputeRunner(
                    core,
                    sequence_length=31,
                    chunk_size=7,
                    n_sequence_outputs=2,
                    pack_spike_checkpoints=True,
                )
                floating = [value for value in initial if value.dtype.is_floating]
                targets = [
                    *floating,
                    cell.recurrent_weight_values,
                    cell.inputs["drive"]["input_weight_values"],
                ]

                def evaluate(segmented):
                    with tf.GradientTape() as tape:
                        tape.watch(floating)
                        outputs = (
                            runner(sequence, initial)
                            if segmented
                            else core([sequence, *initial])
                        )
                        loss = 0.5 * (
                            tf.reduce_mean(tf.cast(outputs[0], tf.float32))
                            + tf.reduce_mean(outputs[1])
                        ) + tf.reduce_mean(tf.cast(outputs[3], tf.float32))
                    return outputs, tape.gradient(loss, targets)

                full, full_gradients = tf.function(lambda: evaluate(False))()
                replay, replay_gradients = tf.function(lambda: evaluate(True))()
                assert full[0].dtype == sequence.dtype
                assert full[1].dtype == tf.float32
                assert np.count_nonzero(full[0][..., 0]) > 0
                assert np.max(full[-1]) > 1
                np.testing.assert_array_equal(full[8], np.full(32, 4095 + 31))
                for actual, expected in zip(replay, full):
                    np.testing.assert_array_equal(actual, expected)
                tolerance = 3e-3 if policy == "mixed_float16" else 1e-5
                for actual, expected in zip(replay_gradients, full_gradients):
                    assert actual is not None and expected is not None
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                for gradient in full_gradients[-2:]:
                    assert np.isfinite(gradient).all() and np.any(gradient != 0)
                results.append(
                    (
                        [value.numpy() for value in full],
                        [value.numpy() for value in full_gradients],
                    )
                )
                cell.close_fused_cuda()
        np.testing.assert_array_equal(
            results[1][0][0][..., 0], results[0][0][0][..., 0]
        )
        np.testing.assert_array_equal(results[1][0][-1], results[0][0][-1])
        for actual, expected in zip(
            tf.nest.flatten(results[1]), tf.nest.flatten(results[0])
        ):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)


def test_nest_internal_noise_matches_cuda_and_cpu():
    from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available

    if not fused_cuda_available():
        pytest.skip("Fused CUDA operators unavailable")
    results = []
    for device, fused in (("/CPU:0", False), ("/GPU:0", True)):
        with tf.device(device):
            cell = make_cell(delay=3.0, internal_noise=True, use_fused_cuda=fused)
            state = list(cell.zero_state(5, tf.float32))
            state[6] = tf.fill((5,), 4095)
            outputs = []
            histories = []
            for step in range(16):
                output, state = cell(tf.zeros((5, 0)), state)
                outputs.append(output.numpy())
                histories.append(state[-1].numpy())
            results.append((np.array(outputs), np.array(histories), state))
            cell.close_fused_cuda()
    expected, actual = results
    assert np.count_nonzero(expected[1]) > 0
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2][6], expected[2][6])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    for expected_state, actual_state in zip(expected[2], actual[2]):
        np.testing.assert_allclose(actual_state, expected_state, rtol=1e-5, atol=1e-5)


def test_internal_noise_uses_original_population_size_and_delayed_history():
    cell = make_cell(delay=3.0, internal_noise=True)
    initial = cell.zero_state(2, tf.float32)
    expected = cell.sample_noise_spikes(2, initial[6], cell.inputs["drive"])
    assert expected.shape == (2, 1)
    _, state = cell(tf.zeros((2, 0)), initial)
    np.testing.assert_array_equal(state[7][:, :1], expected)
    np.testing.assert_array_equal(state[7][:, 1:], [[0.0], [0.0]])
    np.testing.assert_array_equal(state[6], [1, 1])


def test_cached_legacy_state_cannot_silently_drop_input_history(tmp_path):
    from types import SimpleNamespace
    from bmtk.simulator.dpointnet.state_modules.cached_states import CachedInitState

    cell = make_cell(delay=3.0)
    complete, names = cell.zero_state(1, tf.float32, with_names=True)
    filename = tmp_path / "state.npz"
    np.savez(filename, **dict(zip(names, [value.numpy() for value in complete])))
    rnn = SimpleNamespace(cell=cell, batch_size=1, dtype=tf.float32)
    cached = CachedInitState(str(filename), file_type="npz", rnn=rnn)
    state = cached.get_state()
    assert len(state) == 8
    cached._load_fn = lambda *_: complete[:7]
    with pytest.raises(ValueError, match="missing external delay history"):
        cached.get_state()


@pytest.mark.parametrize("return_sequences", [True, False])
def test_symbolic_rnn_preserves_scalar_noise_and_following_history(return_sequences):
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    cell = make_cell(delay=3.0)
    initial, names = cell.zero_state(1, tf.float32, with_names=True)
    values = tf.keras.Input(shape=(None, 1))
    state_inputs = [
        tf.keras.Input(shape=state.shape[1:], dtype=state.dtype, name=name)
        for state, name in zip(initial, names)
    ]
    layer = ExplicitStateRNN(cell, return_sequences=return_sequences, return_state=True)
    layer._autocast = False
    outputs = layer(values, initial_state=state_inputs)
    model = tf.keras.Model([values, *state_inputs], outputs)
    assert len(model.outputs) == 9
    first = model([tf.ones((1, 4, 1)), *initial])
    assert first[7].dtype == tf.int32 and first[8].shape == (1, 2)
    np.testing.assert_array_equal(first[7], [4])
    np.testing.assert_array_equal(first[8], [[1.0, 1.0]])
    late_initial = list(initial)
    late_initial[6] = tf.constant([4095], tf.int32)
    late = model([tf.ones((1, 4, 1)), *late_initial])
    np.testing.assert_array_equal(late[7], [4099])
    second = model([tf.zeros((1, 4, 1)), *first[1:]])
    full = model(
        [tf.concat([tf.ones((1, 4, 1)), tf.zeros((1, 4, 1))], axis=1), *initial]
    )
    for expected, actual in zip(full[1:], second[1:]):
        np.testing.assert_allclose(expected, actual, atol=1e-7)


@pytest.mark.parametrize("dt", [1.0, 0.25])
@pytest.mark.parametrize("tau", [2.0, 10.0, 10.000001, 30.0])
def test_exact_synaptic_integral_matches_matrix_exponential(dt, tau):
    decay, current_factor, current, rise = integration_coefficients(
        dt, [100.0], [10.0], [tau]
    )
    generator = np.array([[-0.1, 0.01, 0], [0, -1 / tau, 1], [0, 0, -1 / tau]])
    expected = expm(dt * generator)
    np.testing.assert_allclose(
        [decay[0], current[0, 0], rise[0, 0]], expected[0], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(current_factor, [(1 - np.exp(-dt / 10)) / 10])


def test_nest_refractory_holds_adaptation_and_blocks_last_refractory_step():
    voltage = tf.constant([[0.7]])
    adaptation = tf.constant([[[-0.1, -0.2]]])
    result = active_update(
        voltage,
        tf.constant([[1]]),
        adaptation,
        tf.zeros((1, 1, 1)),
        tf.zeros((1, 1, 1)),
        decay=0.9,
        current_factor=0.1,
        asc_decay=0.8,
        asc_mean=0.9,
        psc_voltage=1.0,
        rise_voltage=1.0,
        reset_voltage=0.0,
        hard_reset=True,
    )
    assert result[0].numpy()[0, 0] == 0
    assert result[1].numpy()[0, 0] == 0
    assert not result[3].numpy()[0, 0]
    np.testing.assert_array_equal(result[2], adaptation)


def test_reset_applies_refractory_decay_only_to_old_adaptation():
    voltage, remaining, adaptation = spike_reset(
        tf.constant([[1.2]]),
        tf.constant([[0]]),
        tf.constant([[[-2.0, -4.0]]]),
        tf.constant([[1.0]]),
        reset_voltage=0.0,
        refractory_steps=3,
        asc_amplitudes=tf.constant([-1.0, -2.0]),
        asc_refractory_decay=0.5,
        hard_reset=True,
    )
    np.testing.assert_allclose(adaptation, [[[-2.0, -4.0]]])
    assert voltage.numpy()[0, 0] == 0
    assert remaining.numpy()[0, 0] == 3
