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
from bmtk.simulator.dpointnet.custom_ops import (
    build_csr_connectivity,
    fused_cuda_available,
    fused_spike_currents,
    reorder_csr_values,
)
from bmtk.simulator.dpointnet.network_adaptor import lex_sort_order_np
from bmtk.simulator.dpointnet.rnn_model import RNN
from bmtk.simulator.dpointnet.segmented_recompute import (
    _pack_spikes,
    _unpack_spikes,
)
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


def test_segmented_recompute_transforms_accumulated_variable_gradient_once():
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
    transform_calls = []

    def double_variable_gradients(variables, gradients):
        transform_calls.append(tuple(variable.name for variable in variables))
        return tuple(2.0 * gradient for gradient in gradients)

    runner = training.SegmentedRecomputeRunner(
        core_model,
        sequence_length=7,
        chunk_size=3,
        n_sequence_outputs=2,
        variable_gradient_transform=double_variable_gradients,
    )
    values = tf.constant(np.linspace(-0.5, 0.8, 14, dtype=np.float32).reshape(2, 7, 1))
    state = tf.constant([[0.2], [-0.1]], tf.float32)

    with tf.GradientTape() as tape:
        output = core_model((values, state))
        reference_loss = tf.add_n([tf.reduce_sum(value) for value in output])
    reference_gradients = tape.gradient(reference_loss, core_model.trainable_variables)

    with tf.GradientTape() as tape:
        output = runner(values, (state,))
        transformed_loss = tf.add_n([tf.reduce_sum(value) for value in output])
    transformed_gradients = tape.gradient(
        transformed_loss, core_model.trainable_variables
    )

    assert len(transform_calls) == 1
    for transformed, reference in zip(transformed_gradients, reference_gradients):
        np.testing.assert_allclose(transformed, 2.0 * reference, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_segmented_direct_csr_recurrent_gradient_matches_canonical_order():
    indices = np.array([[0, 0], [1, 0], [1, 2], [0, 1]], np.int64)
    synapse_types = np.array([0, 1, 0, 1], np.int64)
    connectivity = build_csr_connectivity(
        indices,
        synapse_types,
        3,
        2,
        2,
        build_compact_pairs=True,
    )
    basis = tf.constant(
        [[1.0, 0.5, 0.25, 0.125], [0.25, 2.0, 0.75, 1.5]],
        tf.float16,
    )

    class CurrentSequence(tf.keras.layers.Layer):
        def __init__(self, write_csr_gradient):
            super().__init__(dtype="mixed_float16")
            self.write_csr_gradient = write_csr_gradient
            self.master = self.add_weight(
                name="recurrent_master",
                shape=(4,),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant([1.0, 2.0, 3.0, 4.0]),
                autocast=False,
            )

        def call(self, values):
            sequence, state = values
            csr_weights = reorder_csr_values(
                tf.cast(self.master, tf.float16), connectivity
            )
            time_major = tf.transpose(sequence, [1, 0, 2])
            currents = tf.map_fn(
                lambda spikes: fused_spike_currents(
                    spikes,
                    self.master,
                    csr_weights,
                    connectivity,
                    basis,
                    n_post=2,
                    compute_spike_gradient=True,
                    use_packed_sm120_backward="auto",
                    write_csr_weight_gradient=self.write_csr_gradient,
                ),
                time_major,
                fn_output_signature=tf.TensorSpec([64, 4], tf.float16),
            )
            currents = tf.reshape(currents, [-1, 32, 2, 4])
            currents = tf.transpose(currents, [1, 0, 2, 3])
            return currents, currents * tf.constant(1.5, tf.float16), state

    def build_model(write_csr_gradient):
        sequence = tf.keras.Input(batch_shape=(32, None, 3), dtype=tf.float16)
        state = tf.keras.Input(batch_shape=(32, 1), dtype=tf.float16)
        layer = CurrentSequence(write_csr_gradient)
        return tf.keras.Model((sequence, state), layer((sequence, state))), layer

    canonical_model, canonical_layer = build_model(False)
    direct_model, direct_layer = build_model(True)
    canonical_runner = training.SegmentedRecomputeRunner(canonical_model, 5, 2, 2)
    direct_cell = SimpleNamespace(
        recurrent_weight_values=direct_layer.master,
        recurrent_fused_connectivity=connectivity,
    )
    direct_runner = training.SegmentedRecomputeRunner(
        direct_model,
        5,
        2,
        2,
        variable_gradient_transform=lambda variables, gradients: (
            GLIF3Cell.restore_segmented_variable_gradients(
                direct_cell, variables, gradients
            )
        ),
    )
    spike_values = np.zeros((32, 5, 3), np.float16)
    spike_values[::2, :, 0] = 1.0
    spike_values[1::3, 1::2, 1] = 2.0
    spike_values[2::5, 2:, 2] = 3.0
    sequence = tf.constant(spike_values)
    state = tf.zeros([32, 1], tf.float16)

    def evaluate(runner, layer):
        with tf.GradientTape() as tape:
            outputs = runner(sequence, (state,))
            loss = tf.reduce_sum(outputs[0]) + tf.reduce_sum(
                outputs[1] * tf.constant(0.25, tf.float16)
            )
        return outputs, tape.gradient(loss, layer.master)

    canonical_outputs, canonical_gradient = evaluate(canonical_runner, canonical_layer)
    direct_outputs, direct_gradient = evaluate(direct_runner, direct_layer)

    for direct, canonical in zip(direct_outputs, canonical_outputs):
        np.testing.assert_array_equal(direct, canonical)
    np.testing.assert_array_equal(direct_gradient, canonical_gradient)


def test_direct_csr_gradient_requires_gradient_checkpointing():
    engine = object.__new__(training.TrainingEngine)
    engine.rnn = SimpleNamespace(
        cell=SimpleNamespace(_use_direct_csr_recurrent_gradient=True)
    )
    engine.gradient_checkpointing = False
    engine._extractor_forward = None

    with pytest.raises(ValueError, match="requires gradient_checkpointing=True"):
        engine.prepare_gradient_checkpointing()


@pytest.mark.parametrize("width", [31, 32, 62])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_spike_checkpoint_pack_roundtrip(width, dtype):
    rng = np.random.default_rng(29)
    patterns = (
        np.zeros((3, width), dtype=np.float32),
        np.ones((3, width), dtype=np.float32),
        rng.integers(0, 2, size=(3, width)).astype(np.float32),
    )

    for pattern in patterns:
        spikes = tf.cast(pattern, dtype)
        packed = _pack_spikes(spikes)
        restored = _unpack_spikes(packed, width, dtype)

        assert packed.dtype == tf.int32
        assert packed.shape == (3, (width + 30) // 31)
        np.testing.assert_array_equal(restored, spikes)


def test_spike_checkpoint_pack_treats_nonzero_values_as_spikes():
    values = tf.constant([[-2.0, 0.0, 0.5, 1.0]], tf.float32)

    restored = _unpack_spikes(_pack_spikes(values), 4, tf.float32)

    np.testing.assert_array_equal(restored, [[1.0, 0.0, 1.0, 1.0]])


def test_packed_segmented_recompute_matches_unpacked_outputs_and_gradients():
    @tf.custom_gradient
    def binary_straight_through(values):
        spikes = tf.cast(values > 0.0, values.dtype)

        def grad(upstream):
            return upstream

        return spikes, grad

    inputs = tf.keras.layers.Input(shape=(None, 3), dtype=tf.float32)
    initial_state = tf.keras.layers.Input(shape=(35,), dtype=tf.float32)
    cell = tf.keras.layers.SimpleRNNCell(
        35,
        activation=binary_straight_through,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(0.1),
        recurrent_initializer=tf.keras.initializers.Constant(0.02),
    )
    sequence, final_state = tf.keras.layers.RNN(
        cell, return_sequences=True, return_state=True
    )(inputs, initial_state=[initial_state])
    core_model = tf.keras.Model(
        (inputs, initial_state),
        (sequence, sequence * tf.constant(1.5), final_state),
    )
    unpacked_runner = training.SegmentedRecomputeRunner(
        core_model,
        sequence_length=7,
        chunk_size=3,
        n_sequence_outputs=2,
        differentiate_inputs=True,
    )
    packed_runner = training.SegmentedRecomputeRunner(
        core_model,
        sequence_length=7,
        chunk_size=3,
        n_sequence_outputs=2,
        differentiate_inputs=True,
        pack_spike_checkpoints=True,
    )
    input_values = (
        np.random.default_rng(31).uniform(-0.2, 0.2, size=(2, 7, 3)).astype(np.float32)
    )
    state_values = np.zeros((2, 35), dtype=np.float32)
    state_values[:, ::3] = 1.0

    def evaluate(runner):
        values = tf.Variable(input_values)
        state = tf.Variable(state_values)
        with tf.GradientTape() as tape:
            outputs = runner(values, (state,))
            loss = (
                tf.reduce_sum(outputs[0] * 0.25)
                + tf.reduce_sum(outputs[1] * 0.5)
                + tf.reduce_sum(outputs[2] * 0.75)
            )
        gradients = tape.gradient(
            loss, (values, state, *core_model.trainable_variables)
        )
        return outputs, gradients

    unpacked_outputs, unpacked_gradients = evaluate(unpacked_runner)
    packed_outputs, packed_gradients = evaluate(packed_runner)

    for packed, unpacked in zip(packed_outputs, unpacked_outputs):
        np.testing.assert_array_equal(packed, unpacked)
    for packed, unpacked in zip(packed_gradients, unpacked_gradients):
        np.testing.assert_allclose(packed, unpacked, rtol=1e-6, atol=1e-6)


def test_training_engine_validates_checkpoint_chunk_size():
    engine = training.TrainingEngine(
        rnn=SimpleNamespace(),
        n_epochs=1,
        steps_per_epoch=1,
        gradient_checkpointing=True,
        gradient_checkpoint_chunk_size=17,
    )
    assert engine.gradient_checkpoint_chunk_size == 17
    assert engine.pack_spike_checkpoints is False

    packed_engine = training.TrainingEngine(
        rnn=SimpleNamespace(),
        n_epochs=1,
        steps_per_epoch=1,
        pack_spike_checkpoints=True,
    )
    assert packed_engine.pack_spike_checkpoints is True

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
