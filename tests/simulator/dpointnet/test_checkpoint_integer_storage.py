import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner


@pytest.mark.parametrize(
    "dtype", [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32, tf.int64]
)
@pytest.mark.parametrize("graph", [False, True])
def test_integer_checkpoint_storage_preserves_extrema_state_and_input_vjp(dtype, graph):
    class Counter(tf.keras.layers.Layer):
        def call(self, values):
            inputs, state = values
            length = tf.shape(inputs)[1]
            offset = tf.range(length, dtype=tf.int64)[None, :, None]
            counter = tf.math.floormod(tf.cast(state[:, None, :], tf.int64) + offset, 8)
            output = inputs * (1 + tf.cast(counter, inputs.dtype))
            final = tf.cast(
                tf.math.floormod(
                    tf.cast(state, tf.int64) + tf.cast(length, tf.int64), 8
                ),
                dtype,
            )
            return output, final

    inputs = tf.keras.Input((None, 1), dtype=tf.float32)
    state = tf.keras.Input((1,), dtype=dtype)
    model = tf.keras.Model((inputs, state), Counter()((inputs, state)))
    runner = SegmentedRecomputeRunner(
        model,
        sequence_length=7,
        chunk_size=3,
        n_sequence_outputs=1,
        differentiate_inputs=True,
    )
    limits = np.iinfo(dtype.as_numpy_dtype)
    initial = tf.constant([[limits.min], [limits.max]], dtype)
    values = tf.reshape(tf.range(14, dtype=tf.float32), (2, 7, 1))

    def evaluate():
        results = []
        for segmented in (False, True):
            with tf.GradientTape() as tape:
                tape.watch(values)
                outputs = (
                    runner(values, (initial,))
                    if segmented
                    else model((values, initial))
                )
                objective = tf.reduce_sum(outputs[0])
            results.append((*outputs, tape.gradient(objective, values)))
        return results

    full, segmented = tf.function(evaluate)() if graph else evaluate()
    for actual, expected in zip(segmented, full):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    assert segmented[1].dtype == dtype
    _, _, history = runner._forward(values, (initial,), None)
    expected_dtype = tf.int32 if dtype.size < 4 else dtype
    assert history[0].dtype == expected_dtype
    np.testing.assert_array_equal(history[0].read(0).numpy(), initial.numpy())
