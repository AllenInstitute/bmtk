import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet import training
from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
    GLIF3Cell,
    calculate_synaptic_currents,
    make_pre_ind_table,
)
from bmtk.simulator.dpointnet.network_adaptor import lex_sort_order_np
from bmtk.simulator.dpointnet.rnn_model import RNN
from bmtk.simulator.dpointnet.optimizers import (
    ExponentiatedAdam,
    LinearWarmupCosineDecay,
    optimizer_supports_loss_scaling,
    scale_loss_for_optimizer,
)
from bmtk.simulator.dpointnet.state_modules.cached_states import CachedInitState
from bmtk.simulator.dpointnet.state_modules.input_state import _complete_noise_state


def test_dpointnet_import_preserves_default_tensorflow_allocator():
    environment = os.environ.copy()
    environment.pop("TF_GPU_ALLOCATOR", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import bmtk.simulator.dpointnet; "
            'print(repr(os.environ.get("TF_GPU_ALLOCATOR")))',
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "None"


def test_refresh_weight_shadows_after_multiple_optimizer_steps():
    recurrent_master = tf.Variable([1.0], dtype=tf.float32, trainable=True)
    recurrent_shadow = tf.Variable([1.0], dtype=tf.float16, trainable=False)
    input_master = tf.Variable([2.0], dtype=tf.float32, trainable=True)
    input_shadow = tf.Variable([2.0], dtype=tf.float16, trainable=False)
    cell = SimpleNamespace(
        recurrent_weight_values=recurrent_master,
        recurrent_weight_values_compute=recurrent_shadow,
        compute_dtype=tf.float16,
        inputs={
            "bkg": {
                "input_weight_values": input_master,
                "input_weight_values_compute": input_shadow,
            }
        },
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)

    for expected_recurrent, expected_input in ((0.75, 1.5), (0.5, 1.0)):
        optimizer.apply_gradients(
            [
                (tf.constant([0.25]), recurrent_master),
                (tf.constant([0.5]), input_master),
            ]
        )
        GLIF3Cell.refresh_recurrent_weight_shadow(cell)

        np.testing.assert_allclose(recurrent_shadow.numpy(), [expected_recurrent])
        np.testing.assert_allclose(input_shadow.numpy(), [expected_input])


def test_fallback_recurrent_dampening_only_scales_spike_gradient():
    recurrent_indices = tf.constant([[0, 0], [1, 0]], dtype=tf.int64)
    dense_shape = (
        tf.constant(2, dtype=tf.int64),
        tf.constant(2, dtype=tf.int64),
    )
    synapse_types = tf.constant([0, 0], dtype=tf.int32)
    basis = tf.constant([[1.0, 0.5]], dtype=tf.float32)
    pre_ind_table = make_pre_ind_table(recurrent_indices, n_source_neurons=2)

    def currents_and_gradients(dampening):
        spikes = tf.Variable([[1.0, 0.0]], dtype=tf.float32)
        master_weights = tf.Variable([2.0, 3.0], dtype=tf.float32)
        compute_weights = tf.constant([2.0, 3.0], dtype=tf.float32)
        with tf.GradientTape() as tape:
            currents = calculate_synaptic_currents(
                spikes,
                recurrent_indices,
                master_weights,
                compute_weights,
                dense_shape,
                basis,
                synapse_types,
                pre_ind_table,
                tf.constant(dampening, dtype=tf.float32),
            )
            loss = tf.reduce_sum(currents)
        spike_gradient, weight_gradient = tape.gradient(loss, (spikes, master_weights))
        return currents.numpy(), spike_gradient.numpy(), weight_gradient.numpy()

    full = currents_and_gradients(1.0)
    dampened = currents_and_gradients(0.25)

    np.testing.assert_allclose(dampened[0], full[0])
    np.testing.assert_allclose(dampened[1], full[1] * 0.25)
    np.testing.assert_allclose(dampened[2], full[2])


def test_noise_step_is_explicit_state_and_replays_poisson_draws():
    cell = object.__new__(GLIF3Cell)
    cell._n_neurons = 2
    cell.max_delay = 1
    cell._n_syn_basis = 2
    cell._refractory_state_dtype = tf.int16
    cell.noise_seed = tf.constant(53, dtype=tf.int64)
    cell.calculate_input_current_from_spikes = lambda spikes, input_net: spikes
    input_net = {
        "input_dense_shape": (2, 64),
        "spike_prob": tf.constant(0.25, dtype=tf.float32),
    }

    state, names = GLIF3Cell.zero_state(
        cell, batch_size=2, dtype=tf.float32, with_names=True
    )
    first = GLIF3Cell.calculate_noise_current(cell, 2, state[-1], input_net)
    replay = GLIF3Cell.calculate_noise_current(cell, 2, state[-1], input_net)
    next_step = GLIF3Cell.calculate_noise_current(cell, 2, state[-1] + 1, input_net)

    assert names[-1] == "noise_step0"
    assert state[-1].dtype == tf.int32
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first.numpy(), next_step.numpy())


def test_segmented_recompute_matches_full_outputs_states_and_gradients():
    inputs = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32)
    initial_state = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    cell = tf.keras.layers.SimpleRNNCell(
        1,
        activation="tanh",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(0.7),
        recurrent_initializer=tf.keras.initializers.Constant(0.4),
    )
    sequence, final_state = tf.keras.layers.RNN(
        cell, return_sequences=True, return_state=True
    )(inputs, initial_state=[initial_state])
    core_model = tf.keras.Model(
        (inputs, initial_state),
        (sequence, sequence * tf.constant(1.5), final_state),
    )
    runner = training.SegmentedRecomputeRunner(
        core_model,
        sequence_length=7,
        chunk_size=3,
        n_sequence_outputs=2,
        differentiate_inputs=True,
    )
    input_values = np.linspace(-0.5, 0.8, 14, dtype=np.float32).reshape(2, 7, 1)
    state_values = np.array([[0.2], [-0.1]], dtype=np.float32)

    def evaluate(run_segmented):
        values = tf.Variable(input_values)
        state = tf.Variable(state_values)
        with tf.GradientTape() as tape:
            if run_segmented:
                outputs = runner(values, (state,))
            else:
                outputs = tuple(core_model((values, state)))
            loss = (
                tf.reduce_sum(outputs[0] * 0.25)
                + tf.reduce_sum(outputs[1] * 0.5)
                + tf.reduce_sum(outputs[2] * 0.75)
            )
        gradients = tape.gradient(
            loss, (values, state, *core_model.trainable_variables)
        )
        return outputs, gradients

    full_outputs, full_gradients = evaluate(False)
    segmented_outputs, segmented_gradients = evaluate(True)

    for segmented, full in zip(segmented_outputs, full_outputs):
        np.testing.assert_allclose(segmented, full, rtol=1e-6, atol=1e-6)
    for segmented, full in zip(segmented_gradients, full_gradients):
        np.testing.assert_allclose(segmented, full, rtol=2e-6, atol=2e-6)


def test_training_engine_validates_checkpoint_chunk_size():
    engine = training.TrainingEngine(
        rnn=SimpleNamespace(),
        n_epochs=1,
        steps_per_epoch=1,
        gradient_checkpointing=True,
        gradient_checkpoint_chunk_size=17,
    )
    assert engine.gradient_checkpoint_chunk_size == 17

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        training.TrainingEngine(
            rnn=SimpleNamespace(),
            n_epochs=1,
            steps_per_epoch=1,
            gradient_checkpoint_chunk_size=0,
        )


def test_parallel_step_slices_state_and_reports_condition_mean():
    captured_states = []

    def loss_function(spikes, model_state, **kwargs):
        captured_states.append(model_state)
        return tf.reduce_mean(spikes)

    weight = tf.Variable(1.0, dtype=tf.float32)
    engine = object.__new__(training.TrainingEngine)
    engine._parameters = [
        SimpleNamespace(
            name="evoked", batch_size=2, loss_functions={"loss": loss_function}
        ),
        SimpleNamespace(
            name="spontaneous", batch_size=2, loss_functions={"loss": loss_function}
        ),
    ]
    engine._batch_indices = None
    engine._inputs_sig_factory = SimpleNamespace(build=lambda targets: targets)
    engine._normalizers = None
    engine._optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    engine.rnn = SimpleNamespace(
        model=SimpleNamespace(trainable_variables=[weight]),
    )

    def run_extractor(inputs, initial_state):
        values = inputs * weight
        row_ids = tf.cast(tf.range(tf.shape(inputs)[0]), tf.float32)[:, None]
        return (values, values), row_ids, row_ids + 10.0

    engine._run_extractor = run_extractor
    engine._prepare_loss_kwargs = lambda parameter, spikes, targets: {}
    inputs = [tf.ones((2, 1)), tf.fill((2, 1), 3.0)]

    loss_values = engine._train_step_parallel(inputs, [{}, {}], init_state=None)

    np.testing.assert_allclose(weight.numpy(), 0.6)
    np.testing.assert_allclose(loss_values["__total_loss"].numpy(), 2.0)
    for actual, expected in zip(captured_states[0], ([0.0, 1.0], [10.0, 11.0])):
        np.testing.assert_allclose(tf.reshape(actual, [-1]), expected)
    for actual, expected in zip(captured_states[1], ([2.0, 3.0], [12.0, 13.0])):
        np.testing.assert_allclose(tf.reshape(actual, [-1]), expected)


def test_series_refreshes_compute_shadow_between_parameter_updates():
    master = tf.Variable(1.0, dtype=tf.float32)
    shadow = tf.Variable(1.0, dtype=tf.float32, trainable=False)
    observed_shadows = []

    class FakeCell:
        def refresh_recurrent_weight_shadow(self):
            shadow.assign(master)

    def loss_function(**kwargs):
        return master

    engine = object.__new__(training.TrainingEngine)
    engine._parameters = [
        SimpleNamespace(name="evoked", loss_functions={"loss": loss_function}),
        SimpleNamespace(name="spontaneous", loss_functions={"loss": loss_function}),
    ]
    engine._normalizers = None
    engine._optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    engine.rnn = SimpleNamespace(
        model=SimpleNamespace(trainable_variables=[master]),
        cell=FakeCell(),
    )

    def run_extractor(inputs, initial_state):
        observed_shadows.append(float(shadow.numpy()))
        return (inputs, inputs), tf.zeros((1, 1))

    engine._run_extractor = run_extractor
    engine._prepare_loss_kwargs = lambda parameter, spikes, targets: {}

    engine._train_step_series(
        [tf.ones((1, 1)), tf.ones((1, 1))], [{}, {}], init_state=None
    )

    np.testing.assert_allclose(observed_shadows, [1.0, 0.9])
    np.testing.assert_allclose(master.numpy(), 0.8)


def test_cached_npz_state_defaults_legacy_noise_step_to_zero(tmp_path):
    state_names = (
        "z0_buf",
        "v0",
        "r0",
        "asc",
        "psc_rise0",
        "psc0",
        "noise_step0",
    )
    arrays = {name: np.ones((2, 1), dtype=np.float32) for name in state_names[:-1]}
    path = Path(tmp_path) / "legacy_state.npz"
    np.savez(path, **arrays)
    cell = SimpleNamespace(
        zero_state=lambda batch_size, dtype, with_names=False: (
            (
                tuple(tf.zeros_like(value) for value in arrays.values())
                + (tf.zeros(2, tf.int32),),
                state_names,
            )
            if with_names
            else tuple(tf.zeros_like(value) for value in arrays.values())
            + (tf.zeros(2, tf.int32),)
        )
    )
    rnn = SimpleNamespace(cell=cell, batch_size=2, dtype=tf.float32)

    state = CachedInitState._load_npz(path, rnn)

    assert len(state) == 7
    np.testing.assert_array_equal(state[-1], np.zeros(2, dtype=np.int32))


def test_state_only_rollout_completes_omitted_noise_step():
    initial_state = tuple(tf.zeros((2, 1)) for _ in range(6)) + (
        tf.constant([3, 3], dtype=tf.int32),
    )
    state_out = tuple(tf.ones((2, 1)) for _ in range(6))

    completed = _complete_noise_state(state_out, initial_state, sequence_length=5)

    assert len(completed) == 7
    np.testing.assert_array_equal(completed[-1], [8, 8])


def test_training_refreshes_weight_shadows_after_each_step(monkeypatch):
    class FakeDataIterator:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeCell:
        def __init__(self):
            self.refresh_count = 0

        def refresh_recurrent_weight_shadow(self):
            self.refresh_count += 1

    callbacks = SimpleNamespace(
        on_train_begin=lambda: None,
        on_epoch_start=lambda: None,
        on_step_start=lambda: None,
        on_step_end=lambda loss: None,
        on_epoch_end=lambda loss: False,
        on_train_end=lambda **kwargs: None,
    )
    cell = FakeCell()
    engine = object.__new__(training.TrainingEngine)
    engine._parameters = [SimpleNamespace(input_generators=[], batch_size=1, seq_len=2)]
    engine.rnn = SimpleNamespace(ordered_inputs_populations=[], _cell=cell)
    engine.regenerate_initial_state_each_epoch = False
    engine._init_state_mod = SimpleNamespace(get_state=lambda: None)
    engine._prepare_normalizers = lambda: None
    engine._normalizers = None
    engine.gradient_checkpointing = False
    engine._extractor_forward = None
    engine._callbacks = callbacks
    engine.n_epochs = 1
    engine.steps_per_epoch = 3
    engine._training_approach = "single"
    engine._next_spikes_with_retry = lambda input_itr: ("spikes", "targets")
    engine._distributed_train_step = lambda spikes, targets, init_state: {
        "loss": tf.constant(0.0)
    }
    engine._distributed_validation_step = lambda *args, **kwargs: tf.constant(0.0)
    monkeypatch.setattr(training, "DataIterator", FakeDataIterator)

    engine.train()

    assert cell.refresh_count == engine.steps_per_epoch


def test_lex_sort_order_avoids_integer_overflow():
    indices = np.array(
        [
            [np.iinfo(np.uint32).max, np.iinfo(np.uint32).max],
            [np.iinfo(np.uint32).max - 1, np.iinfo(np.uint32).max],
            [0, 0],
        ],
        dtype=np.uint32,
    )

    order = lex_sort_order_np(indices)

    np.testing.assert_array_equal(order, [2, 1, 0])


def test_rnn_wraps_float16_optimizer_with_loss_scaling():
    class FakeTrainingEngine:
        def __init__(self):
            self.optimizer = tf.keras.optimizers.SGD()
            self.learning_rule = SimpleNamespace(
                build=lambda rnn: None,
                uses_bptt=True,
            )
            self.trained = False

        def set_optimizer(self, optimizer):
            self.optimizer = optimizer

        def train(self):
            self.trained = True

    rnn = object.__new__(RNN)
    rnn.extractor_model = object()
    rnn.strategy = tf.distribute.get_strategy()
    rnn.dtype = tf.float16
    rnn.model = SimpleNamespace(trainable_variables=[])
    engine = FakeTrainingEngine()

    rnn.train(engine)

    assert isinstance(engine.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
    assert engine.trained


def test_base_optimizer_scale_loss_method_is_not_active_loss_scaling():
    optimizer = ExponentiatedAdam(learning_rate=0.005)

    assert not optimizer_supports_loss_scaling(optimizer)
    np.testing.assert_allclose(
        scale_loss_for_optimizer(optimizer, tf.constant(1.0)).numpy(),
        1.0,
    )


def test_paper_protocol_warmup_cosine_schedule_endpoints():
    schedule = LinearWarmupCosineDecay(
        warmup_start_lr=0.0005,
        warmup_target_lr=0.0126,
        warmup_steps=30,
        cosine_steps=555,
        min_lr=0.0005,
    )

    np.testing.assert_allclose(schedule(0), 0.0005)
    np.testing.assert_allclose(schedule(29), 0.0126)
    np.testing.assert_allclose(schedule(30), 0.0126)
    np.testing.assert_allclose(schedule(585), 0.0005)
    np.testing.assert_allclose(schedule(586), 0.0005)


def test_loss_scale_optimizer_is_active_loss_scaling():
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        ExponentiatedAdam(learning_rate=0.005)
    )

    assert optimizer_supports_loss_scaling(optimizer)
    assert scale_loss_for_optimizer(optimizer, tf.constant(1.0)).numpy() > 1.0


def test_loss_scaling_preserves_small_gradient_through_float16_activation():
    activation = tf.Variable(1.0, dtype=tf.float16)
    with tf.GradientTape() as tape:
        loss = tf.cast(activation, tf.float32) * tf.constant(1.0e-8)
    unscaled_gradient = tape.gradient(loss, activation)

    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        ExponentiatedAdam(learning_rate=0.005)
    )
    with tf.GradientTape() as tape:
        loss = tf.cast(activation, tf.float32) * tf.constant(1.0e-8)
        scaled_loss = scale_loss_for_optimizer(optimizer, loss)
    scaled_gradient = tape.gradient(scaled_loss, activation)

    assert unscaled_gradient.numpy() == 0.0
    assert scaled_gradient.numpy() > 0.0
