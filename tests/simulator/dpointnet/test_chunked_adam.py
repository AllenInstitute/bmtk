"""ChunkedAdam equals Keras Adam bit for bit while updating in bounded chunks."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.cell_models.glif3_cell import SignedConstraint
from bmtk.simulator.dpointnet.optimizers import ChunkedAdam, _keras_major_version, create_optimizer

pytestmark = pytest.mark.skipif(_keras_major_version() < 3, reason="chunked_adam requires Keras 3")


def _setup(seed, constrained):
    rng = np.random.default_rng(seed)
    big = rng.normal(size=1003).astype(np.float32)
    small = rng.normal(size=5).astype(np.float32)
    condition = tf.constant(big >= 0) if constrained else None
    def make(value, name, constraint=None):
        variable = tf.keras.Variable(value, name=name)
        variable.constraint = constraint
        return variable

    variables = [make(big, "big", SignedConstraint(condition) if constrained else None),
                 make(small, "small")]
    gradients = [[rng.normal(size=v.shape[0]).astype(np.float32) * 5 for v in variables] for _ in range(4)]
    return variables, gradients


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_chunked_adam_matches_adam(constrained, compiled):
    results = []
    for chunked in (False, True):
        variables, gradients = _setup(3, constrained)
        optimizer = (ChunkedAdam(learning_rate=0.05, epsilon=1e-7, update_chunk_size=128)
                     if chunked else tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=1e-7))

        def step(grads):
            optimizer.apply_gradients(zip(grads, variables))

        step = tf.function(step) if compiled else step
        for grads in gradients:
            step([tf.constant(g) for g in grads])
        slots = [np.asarray(v.numpy()) for v in optimizer.variables[1:]]
        results.append(([v.numpy() for v in variables], slots, int(optimizer.iterations.numpy())))
    (plain, plain_slots, plain_it), (chunk, chunk_slots, chunk_it) = results
    assert plain_it == chunk_it == 4
    for a, b in zip(plain, chunk):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(plain_slots, chunk_slots):
        np.testing.assert_array_equal(a, b)
    if constrained:
        variables, _ = _setup(3, True)
        positive = variables[0].constraint.condition.numpy()
        assert np.all(chunk[0][positive] >= 0) and np.all(chunk[0][~positive] <= 0)


def test_chunked_adam_keeps_constraint_after_apply():
    variables, gradients = _setup(4, True)
    optimizer = ChunkedAdam(learning_rate=0.05, update_chunk_size=100)
    constraint = variables[0].constraint
    optimizer.apply_gradients(zip([tf.constant(g) for g in gradients[0]], variables))
    assert variables[0].constraint is constraint


def test_chunked_adam_matches_adam_under_mirrored_strategy():
    # The strategy hands update_step per-replica components, not the Keras variables.
    strategy = tf.distribute.MirroredStrategy(["/cpu:0"])
    rng = np.random.default_rng(5)
    initial = rng.normal(size=1003).astype(np.float32)
    gradients = [tf.constant(rng.normal(size=1003).astype(np.float32) * 5) for _ in range(4)]

    class Weights(tf.keras.layers.Layer):
        def build(self):
            self.w = self.add_weight(shape=(1003,), initializer=tf.keras.initializers.Constant(initial),
                                     constraint=SignedConstraint(tf.constant(initial >= 0)))

    results = []
    for chunked in (False, True):
        with strategy.scope():
            layer = Weights()
            layer.build()
            optimizer = (ChunkedAdam(learning_rate=0.05, epsilon=1e-7, update_chunk_size=128)
                         if chunked else tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=1e-7))

        for gradient in gradients:
            tf.function(lambda: strategy.run(lambda: optimizer.apply_gradients([(gradient, layer.w)])))()
        results.append(layer.w.numpy())
    np.testing.assert_array_equal(results[0], results[1])
    positive = initial >= 0
    assert np.all(results[1][positive] >= 0) and np.all(results[1][~positive] <= 0)
    assert np.any(results[1] == 0)


def test_create_optimizer_builds_chunked_adam():
    optimizer = create_optimizer("chunked_adam", 0.01, {"name": "chunked_adam", "update_chunk_size": 64})
    assert isinstance(optimizer, ChunkedAdam)
    assert optimizer.update_chunk_size == 64
    assert optimizer.epsilon == pytest.approx(1e-11)
    with pytest.raises(ValueError, match="update_chunk_size"):
        ChunkedAdam(update_chunk_size=0)
    with pytest.raises(ValueError, match="amsgrad"):
        ChunkedAdam(amsgrad=True)
