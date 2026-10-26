import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.loss_functions.weight_regularization import (
    _grouped_emd_custom_gradient,
)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_batched_emd_preserves_group_reductions_and_canonical_ties(dtype, graph, ties):
    rng = np.random.default_rng(41)
    size = 53
    order = tf.constant(rng.permutation(size), tf.uint32)
    slices = ((0, 1), (1, 17), (17, 22), (22, size))
    initial = tf.constant(
        np.concatenate([np.sort(rng.normal(size=end - start)) for start, end in slices]),
        dtype,
    )
    current = rng.integers(-2, 3, size) if ties else rng.normal(size=size)
    values = tf.constant(current, dtype)
    denominators = tuple((end - start) * len(slices) for start, end in slices)

    def compare():
        outputs = []
        for strategy in ("per_group", "batched"):
            with tf.GradientTape() as tape:
                tape.watch(values)
                loss = _grouped_emd_custom_gradient(
                    values, order, initial, slices, denominators, strategy
                )
            outputs.append((loss, tape.gradient(loss, values)))
        return outputs

    reference, batched = tf.function(compare)() if graph else compare()
    for expected, actual in zip(reference, batched):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_batched_emd_rejects_unknown_edge_count():
    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def evaluate(values):
        return _grouped_emd_custom_gradient(
            values, tf.constant([0, 1], tf.uint32), tf.constant([1.0, 2.0]),
            ((0, 2),), (2,), "batched",
        )

    with pytest.raises(ValueError, match="static edge count"):
        evaluate.get_concrete_function()
