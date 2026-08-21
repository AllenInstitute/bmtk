from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet import training
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.network_adaptor import lex_sort_order_np
from bmtk.simulator.dpointnet.optimizers import (
    ExponentiatedAdam,
    optimizer_supports_loss_scaling,
    scale_loss_for_optimizer,
)


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
        optimizer.apply_gradients([
            (tf.constant([0.25]), recurrent_master),
            (tf.constant([0.5]), input_master),
        ])
        GLIF3Cell.refresh_recurrent_weight_shadow(cell)

        np.testing.assert_allclose(recurrent_shadow.numpy(), [expected_recurrent])
        np.testing.assert_allclose(input_shadow.numpy(), [expected_input])


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
    engine._parameters = [
        SimpleNamespace(input_generators=[], batch_size=1, seq_len=2)
    ]
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
    engine._distributed_train_step = (
        lambda spikes, targets, init_state: {"loss": tf.constant(0.0)}
    )
    engine._distributed_validation_step = lambda *args, **kwargs: tf.constant(0.0)
    monkeypatch.setattr(training, "DataIterator", FakeDataIterator)

    engine.train()

    assert cell.refresh_count == engine.steps_per_epoch


def test_lex_sort_order_avoids_uint32_overflow():
    indices = np.array(
        [[65535, 65535], [65536, 0], [0, 1]], dtype=np.uint32
    )

    order = lex_sort_order_np(indices)

    np.testing.assert_array_equal(order, [2, 0, 1])


def test_base_optimizer_scale_loss_method_is_not_active_loss_scaling():
    optimizer = ExponentiatedAdam(learning_rate=0.005)

    assert not optimizer_supports_loss_scaling(optimizer)
    np.testing.assert_allclose(
        scale_loss_for_optimizer(optimizer, tf.constant(1.0)).numpy(),
        1.0,
    )


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
