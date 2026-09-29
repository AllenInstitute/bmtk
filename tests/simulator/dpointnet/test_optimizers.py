"""Factory safeguards and actual updates on both Keras 2 and Keras 3."""

import itertools

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.optimizers import (
    ExponentiatedAdam,
    create_optimizer,
    scale_loss_for_optimizer,
    unscale_gradients_for_optimizer,
)


OPTIMIZERS = ("adam", "exp_adam", "sgd")
CLIPPING_MODES = ("clipnorm", "clipvalue", "global_clipnorm")


def _params(name):
    return {"epsilon": 0.1} if name != "sgd" else {"momentum": 0.3, "nesterov": True}


def _reference_gradients(gradients, mode, threshold=5.0):
    if mode == "clipvalue":
        return [np.clip(g, -threshold, threshold) for g in gradients]
    if mode == "clipnorm":
        return [g * min(1.0, threshold / np.linalg.norm(g)) for g in gradients]
    if mode == "global_clipnorm":
        norm = np.sqrt(sum(np.sum(g * g) for g in gradients))
        return [g * min(1.0, threshold / norm) for g in gradients]
    return gradients


def _variables(optimizer):
    variables = optimizer.variables
    return variables() if callable(variables) else variables


@pytest.mark.parametrize("name", OPTIMIZERS)
@pytest.mark.parametrize("mode", (None,) + CLIPPING_MODES)
@pytest.mark.parametrize("loss_scaling", (False, True))
@pytest.mark.parametrize("graph", (False, True))
def test_factory_update_matches_independently_clipped_unscaled_reference(
    name, mode, loss_scaling, graph
):
    physical_gradients = [
        np.array([3000.0, -4000.0], dtype=np.float32),
        np.array([12000.0, 0.5], dtype=np.float32),
    ]
    expected_gradients = _reference_gradients(physical_gradients, mode)
    params = _params(name)
    if mode is not None:
        params[mode] = 5.0
    inner = create_optimizer(name, 0.1, params)
    reference = create_optimizer(name, 0.1, _params(name))
    optimizer = inner
    if loss_scaling:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            inner, initial_scale=128.0, dynamic_growth_steps=1000
        )
    actual_weights = [tf.Variable([2.0, -3.0]), tf.Variable([4.0, -5.0])]
    expected_weights = [tf.Variable(w.numpy()) for w in actual_weights]

    def update():
        with tf.GradientTape() as tape:
            loss = tf.add_n([
                tf.reduce_sum(w * g) for w, g in zip(actual_weights, physical_gradients)
            ])
            scaled_loss = scale_loss_for_optimizer(optimizer, loss)
        gradients = tape.gradient(scaled_loss, actual_weights)
        # Keras 2 unscales explicitly; Keras 3's wrapper unscales during apply.
        gradients = unscale_gradients_for_optimizer(optimizer, gradients)
        optimizer.apply_gradients(zip(gradients, actual_weights))

    (tf.function(update) if graph else update)()
    reference.apply_gradients(zip(map(tf.constant, expected_gradients), expected_weights))

    for actual, expected in zip(actual_weights, expected_weights):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=2e-7)
    actual_slots, expected_slots = _variables(inner), _variables(reference)
    assert len(actual_slots) == len(expected_slots)
    for actual, expected in zip(actual_slots, expected_slots):
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=2e-7)
    assert int(inner.iterations.numpy()) == 1
    if mode is not None:
        assert any(not np.array_equal(g, e) for g, e in zip(
            physical_gradients, expected_gradients
        ))


@pytest.mark.parametrize("name", OPTIMIZERS)
@pytest.mark.parametrize("mode", CLIPPING_MODES)
def test_clipping_config_roundtrip(name, mode):
    params = {"name": name, mode: 5.0, **_params(name)}
    original = dict(params)
    optimizer = create_optimizer(name, 0.1, params)
    restored = type(optimizer).from_config(optimizer.get_config())
    assert params == original
    assert restored.get_config()[mode] == 5.0
    weights = [tf.Variable([2.0, -3.0]) for _ in range(2)]
    for opt, weight in zip((optimizer, restored), weights):
        opt.apply_gradients([(tf.constant([3000.0, -4000.0]), weight)])
    np.testing.assert_allclose(weights[0].numpy(), weights[1].numpy(), rtol=1e-6)


@pytest.mark.parametrize("name", OPTIMIZERS)
@pytest.mark.parametrize("mode", CLIPPING_MODES)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf"),
                                 True, False, "5", [], {}])
def test_invalid_clipping_threshold_rejected(name, mode, value):
    with pytest.raises(ValueError, match=mode + " must be a finite positive"):
        create_optimizer(name, 0.1, {mode: value})


@pytest.mark.parametrize("name", OPTIMIZERS)
@pytest.mark.parametrize("modes", list(itertools.combinations(CLIPPING_MODES, 2)))
def test_conflicting_clipping_modes_rejected(name, modes):
    with pytest.raises(ValueError, match="Only one"):
        create_optimizer(name, 0.1, dict.fromkeys(modes, 5.0))


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_omitted_and_null_clipping_preserve_existing_defaults(name):
    constructor = {
        "adam": tf.keras.optimizers.Adam,
        "exp_adam": ExponentiatedAdam,
        "sgd": tf.keras.optimizers.SGD,
    }[name]
    defaults = {"epsilon": 1e-11} if name != "sgd" else {"momentum": 0.0, "nesterov": False}
    reference = constructor(learning_rate=0.1, **defaults)
    for params in (None, {}, {"name": name}, dict.fromkeys(CLIPPING_MODES)):
        optimizer = create_optimizer(name, 0.1, params)
        for key in (*defaults, *CLIPPING_MODES):
            assert optimizer.get_config()[key] == reference.get_config()[key]
        actual, expected = tf.Variable([2.0, -3.0]), tf.Variable([2.0, -3.0])
        fresh_reference = constructor(learning_rate=0.1, **defaults)
        for opt, weight in ((optimizer, actual), (fresh_reference, expected)):
            opt.apply_gradients([(tf.constant([3000.0, -4000.0]), weight)])
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6)


@pytest.mark.parametrize("name", OPTIMIZERS)
@pytest.mark.parametrize("field", ["global_clip_norm", "learning_rate", "loss_scale_factor",
                                 "unsupported"])
def test_unsupported_fields_are_not_silently_ignored(name, field):
    with pytest.raises(ValueError, match="Unsupported optimizer_params.*" + field):
        create_optimizer(name, 0.1, {field: 1.0})


@pytest.mark.parametrize("name,field", [("adam", "momentum"), ("exp_adam", "nesterov"),
                                       ("sgd", "epsilon")])
def test_optimizer_specific_fields_rejected_for_other_optimizers(name, field):
    with pytest.raises(ValueError, match="Unsupported optimizer_params.*" + field):
        create_optimizer(name, 0.1, {field: 0.1})


def test_selector_and_parameter_type_errors():
    with pytest.raises(ValueError, match="Invalid optimizer"):
        create_optimizer("unknown", 0.1)
    with pytest.raises(ValueError, match="name must match"):
        create_optimizer("adam", 0.1, {"name": "sgd"})
    for params in ([], "", 1):
        with pytest.raises(TypeError, match="must be a mapping"):
            create_optimizer("adam", 0.1, params)


def test_optimizer_instance_is_preserved_but_cannot_ignore_parameters():
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, clipvalue=5.0)
    assert create_optimizer(optimizer, 0.5) is optimizer
    assert create_optimizer(optimizer, 0.5, {}) is optimizer
    with pytest.raises(ValueError, match="cannot configure an optimizer instance"):
        create_optimizer(optimizer, 0.5, {"clipvalue": 1.0})


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_learning_rate_schedule_with_clipping(name):
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, 10, 0.5)
    optimizer = create_optimizer(name, schedule, {"global_clipnorm": 5.0})
    assert optimizer._learning_rate is schedule
