"""Original-projection replay, including deliberately non-repeatable forwards."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from test_precision_credit import make_cell
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import fused_glif_state_available


class ChangingProjectionCell(GLIF3Cell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_calls = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _project_step_currents(self, inputs, states, noise_seed=None):
        currents, history = super()._project_step_currents(inputs, states, noise_seed)
        serial = self.projection_calls.assign_add(1)
        return currents + tf.cast(serial, currents.dtype) * 0.125, history


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def changing_cell(mode):
    network, inputs, options = make_cell(
        mode, return_spec=True, temporal_gradient_precision="float32",
        temporal_checkpoint_chunk_size=3,
    )
    return ChangingProjectionCell(network, inputs, **options)


def variables(cell):
    return tuple(
        v.value if not callable(getattr(v, "value", None)) else v
        for v in cell.trainable_variables
    )


def assert_cache_replay(runner, x, initial, cache):
    sequences, _, boundaries, saved, seed, currents, projection_values = cache
    mismatches = []
    for index in range((int(x.shape[1]) + runner.chunk_size - 1) // runner.chunk_size):
        start, stop = boundaries[index], boundaries[index + 1]
        state = tuple(array.read(index) for array in saved)
        chunk = runner._read_current_chunk(currents, index)
        for shadow in (False, True):
            replay_state = tuple(
                tf.cast(v, tf.float32) if shadow and v.dtype.is_floating else v for v in state
            )
            output, final = runner._loop(
                tf.cast(x[:, start:stop], tf.float32) if shadow else x[:, start:stop],
                replay_state, replay=shadow, recorded_currents=chunk,
                projection_context=runner.cell._prepare_adjoint_projection_context(projection_values) if shadow else None,
                noise_seed=seed,
            )
            expected = tuple(value[:, start:stop] for value in sequences) + tuple(
                array.read(index + 1) for array in saved
            )
            actual = output + tuple(final)
            mismatches.extend(
                tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(a, b.dtype), b), tf.int32))
                for a, b in zip(expected, actual)
            )
    return tf.stack(mismatches)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_original_tape_not_reconstructed_projection_and_full_tape_oracle(mode):
    cell = changing_cell(mode)
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    x = tf.ones((2, 9, 2), tf.float32) * 0.7
    initial = cell.zero_state(2, cell.compute_dtype)
    cache = runner._forward(x, initial)
    assert int(cell.projection_calls.numpy()) == 9
    assert "CPU:0" in cache[5].read(0).device
    np.testing.assert_array_equal(assert_cache_replay(runner, x, initial, cache), 0)
    assert int(cell.projection_calls.numpy()) == 9
    final_gradients = [None] * len(initial)
    final_gradients[1] = 2 * tf.cast(cache[1][1], tf.float32)
    dx, _, dw, _ = runner._backward(
        x, initial, cache, (None, None), final_gradients, variables(cell)
    )
    # Independent unsegmented FP32 tape on the same recorded original primals.
    # Concatenation is confined to this nine-step, two-neuron test oracle.
    original_currents = tf.concat([cache[5].read(i) for i in range(3)], axis=0)
    state32 = tuple(tf.cast(v, tf.float32) if v.dtype.is_floating else v for v in initial)
    with tf.GradientTape() as tape:
        tape.watch(x)
        context = cell._prepare_adjoint_projection_context()
        _, final = runner._loop(
            x, state32, replay=True, recorded_currents=original_currents,
            projection_context=context, noise_seed=cache[4],
        )
        loss = tf.reduce_sum(tf.square(final[1]))
    expected = tape.gradient(loss, (x, *variables(cell)))
    np.testing.assert_allclose(dx, expected[0], rtol=1e-5, atol=1e-6)
    for actual, reference in zip(dw, expected[1:]):
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    assert np.any(dx.numpy() != 0)
    assert int(cell.projection_calls.numpy()) == 9


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("graph", [False, True])
def test_multiple_forward_tapes_are_independent_and_reusable(mode, graph):
    cell = changing_cell(mode)
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    x = tf.ones((2, 9, 2), tf.float32) * 0.7
    initial = cell.zero_state(2, cell.compute_dtype)

    def evaluate():
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            _, first = runner(x, initial)
            _, second = runner(x, initial)
            a, b = tf.reduce_sum(first[1] ** 2), tf.reduce_sum(second[1] ** 2)
            total = a + b
        ga = tape.gradient(a, x)
        gb = tape.gradient(b, x)
        both = tape.gradient(total, x)
        return a, b, ga, gb, both

    a, b, ga, gb, both = (tf.function(evaluate) if graph else evaluate)()
    assert a.numpy() != b.numpy()
    np.testing.assert_allclose(both, ga + gb, rtol=1e-6, atol=1e-6)
    assert int(cell.projection_calls.numpy()) == 18


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_recorded_values_survive_weight_changes_and_strict_checkpoint_restore(mode, tmp_path):
    cell = changing_cell(mode)
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    x = tf.ones((2, 9, 2), tf.float32) * 0.7
    initial = cell.zero_state(2, cell.compute_dtype)
    cache = runner._forward(x, initial)
    final_gradients = [None] * len(initial)
    final_gradients[1] = 2 * cache[1][1]

    def reverse():
        result = runner._backward(
            x, initial, cache, (None, None), final_gradients, variables(cell)
        )
        return (result[0], *result[2])

    expected = reverse()
    optimizer = tf.keras.optimizers.Adam(0.01)
    optimizer.build(cell.trainable_variables)
    checkpoint = tf.train.Checkpoint(cell=cell, optimizer=optimizer)
    checkpoint.save_counter.assign(0)
    prefix = checkpoint.write(str(tmp_path / "checkpoint"))
    masters = [v.numpy().copy() for v in cell.trainable_variables]
    for value in cell.trainable_variables:
        value.assign(value * 1.7)
    cell.refresh_recurrent_weight_shadow()
    for actual, reference in zip(reverse(), expected):
        np.testing.assert_array_equal(actual, reference)
    optimizer.iterations.assign_add(7)
    checkpoint.read(prefix).assert_consumed()
    cell.refresh_recurrent_weight_shadow()
    assert int(optimizer.iterations.numpy()) == 0
    for value, original in zip(cell.trainable_variables, masters):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(assert_cache_replay(runner, x, initial, cache), 0)
    assert int(cell.projection_calls.numpy()) == 9


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("graph", [False, True])
def test_high_fanin_fused_original_current_replay_is_bitwise(mode, graph):
    if not fused_glif_state_available():
        pytest.skip("CUDA operators unavailable")
    rng = np.random.default_rng(20260925)
    count, fanin = 512, 32
    network, inputs, options = make_cell(
        mode, return_spec=True, fused=True, use_fused_cuda=True,
        temporal_gradient_precision="float32", batch_size=32,
        use_packed_sm120_backward=False, use_packed_sm120_external_backward=False,
    )
    network.update(n_nodes=count, node_type_ids=np.arange(count) % 2)
    network["synapses"].update(
        indices=np.column_stack((np.repeat(np.arange(count), fanin),
                                 rng.integers(0, count, count * fanin))).astype(np.uint32),
        weights=rng.normal(0, 20, count * fanin).astype(np.float32),
        delays=np.full(count * fanin, 2.0), dense_shape=(count, count),
        syn_ids=np.zeros(count * fanin, np.int64),
    )
    inputs["drive"].update(
        n_inputs=count, indices=np.column_stack((np.arange(count), np.arange(count))),
        weights=np.full(count, 1000.0), delays=np.ones(count),
        syn_ids=np.zeros(count, np.int64),
    )
    cell = GLIF3Cell(network, inputs, **options)
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    initial = list(cell.zero_state(32, cell.compute_dtype))
    initial[0] = tf.constant(rng.integers(0, 2, initial[0].shape), tf.float16)
    x = tf.ones((32, 9, count), tf.float32) * 0.7

    def evaluate():
        cache = runner._forward(x, initial)
        return assert_cache_replay(runner, x, initial, cache)

    if graph:
        compiled = tf.function(evaluate)
        mismatches = compiled()
        definitions = compiled.get_concrete_function().graph.as_graph_def().library.function
        copies = [
            node for function in definitions for node in function.node_def
            if "original_current_tape_copy" in node.name
        ]
        assert copies and all("CPU:0" in node.device for node in copies)
    else:
        mismatches = evaluate()
    np.testing.assert_array_equal(mismatches, 0)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("fused", [False, True])
def test_public_static_sequence_metadata_for_orientation_loss(mode, fused):
    from test_temporal_input_contract import model
    from bmtk.simulator.dpointnet.loss_functions.orientation_selectivity_loss import OrientationSelectivityLoss

    if fused and not fused_glif_state_available():
        pytest.skip("CUDA operators unavailable")
    rnn = model(mode, fused)
    loss = object.__new__(OrientationSelectivityLoss)
    loss._use_ema_normalizer = True
    loss._method = "crowd_osi"
    loss._pre_delay = 1
    loss._post_delay = 1
    loss._ema_decay = tf.constant(0.95, tf.float32)
    normalizers = {"v1_ema": tf.Variable(tf.zeros(2), trainable=False)}

    @tf.function
    def step(x):
        with tf.GradientTape() as tape:
            output = rnn.run_extractor(x, rnn.zero_state)[0]
            assert output[0].shape[1] == 9
            loss.update_normalizers(tf.stop_gradient(output[0]), normalizers)
            objective = tf.reduce_sum(output[1])
        gradients = tape.gradient(objective, rnn.model.trainable_variables)
        return gradients, normalizers["v1_ema"]

    try:
        gradients, ema = step(tf.ones((2, 9, 2), tf.bool))
        assert all(g is not None and np.all(np.isfinite(g)) for g in gradients)
        assert np.all(np.isfinite(ema))
    finally:
        rnn.cleanup()
