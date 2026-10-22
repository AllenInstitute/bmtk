"""Optimizer checkpoints of CSR-ordered masters: restore order and supported optimizers."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet import csr_master, optimizers, training
from bmtk.simulator.dpointnet.rnn_model import RNN
from test_default_acceleration_options import (
    build_model, canonical_slots, checkpoint, forward_free, restore_policy, step,
)

import keras

# Keras 3 checkpoints named-input masters only through the optimizer's variable
# list, which optimizer.build replaces: restoring before the build does not restore
# them in either layout (unchanged from develop). TF-Keras 2 tracks them in the cell.
KERAS3 = keras.__version__.startswith("3")


def _keras(name):
    return getattr(tf.keras.optimizers, name, None)


def _factories():
    factories = {
        "adam": lambda: tf.keras.optimizers.Adam(0.01, epsilon=1e-11),
        "adam_amsgrad": lambda: tf.keras.optimizers.Adam(0.01, epsilon=1e-11, amsgrad=True),
        "adam_ema": lambda: tf.keras.optimizers.Adam(0.01, epsilon=1e-11, use_ema=True),
        "adamw": lambda: tf.keras.optimizers.AdamW(0.01, epsilon=1e-11),
        "exp_adam": lambda: optimizers.ExponentiatedAdam(learning_rate=0.01, jit_compile=False),
        "exp_adam_amsgrad": lambda: optimizers.ExponentiatedAdam(
            learning_rate=0.01, amsgrad=True, jit_compile=False,
        ),
        "create_adam": lambda: optimizers.create_optimizer("adam", 0.01),
        "create_sgd": lambda: optimizers.create_optimizer("sgd", 0.01),
        "sgd_momentum": lambda: tf.keras.optimizers.SGD(0.01, momentum=0.9),
        "sgd_nesterov": lambda: tf.keras.optimizers.SGD(0.01, momentum=0.9, nesterov=True),
        "rmsprop": lambda: tf.keras.optimizers.RMSprop(0.01),
        "rmsprop_centered_momentum": lambda: tf.keras.optimizers.RMSprop(
            0.01, momentum=0.5, centered=True,
        ),
        "adagrad": lambda: tf.keras.optimizers.Adagrad(0.01),
        "adamax": lambda: tf.keras.optimizers.Adamax(0.01),
        "nadam": lambda: tf.keras.optimizers.Nadam(0.01),
        "adadelta": lambda: tf.keras.optimizers.Adadelta(0.5),
        "ftrl": lambda: tf.keras.optimizers.Ftrl(0.01),
        "loss_scale_adam": lambda: tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(0.01, epsilon=1e-11),
        ),
        "loss_scale_rmsprop_centered": lambda: tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.RMSprop(0.01, momentum=0.5, centered=True),
        ),
    }
    for name in ("Lion", "Lamb"):
        if _keras(name) is not None:
            factories[name.lower()] = lambda name=name: _keras(name)(0.01)
    if hasattr(tf.keras.optimizers.Optimizer, "add_optimizer_variables"):
        factories["adam_gradient_accumulation"] = lambda: tf.keras.optimizers.Adam(
            0.01, epsilon=1e-11, gradient_accumulation_steps=2,
        )
    legacy = getattr(tf.keras.optimizers, "legacy", None)
    if legacy is not None and hasattr(legacy, "Optimizer") and hasattr(legacy.Adam, "get_slot_names"):
        try:
            legacy.Adam(0.01)
        except Exception:
            legacy = None
        if legacy is not None:
            factories["legacy_adam"] = lambda: legacy.Adam(0.01, epsilon=1e-11)
            factories["legacy_sgd_momentum"] = lambda: legacy.SGD(0.01, momentum=0.9)
            factories["legacy_rmsprop_centered"] = lambda: legacy.RMSprop(
                0.01, momentum=0.5, centered=True,
            )
    return factories


FACTORIES = _factories()
# Elementwise updates continue bit-identically across layouts; Lamb's trust ratio
# reduces each variable's norm, whose summation order follows the live layout.
REDUCING = {"lamb"}
# TF-Keras 2 Nadam keeps its momentum-schedule product outside the checkpoint, so
# every resume (also canonical to canonical) continues differently from the source.
UNCHECKPOINTED_STATE = set() if KERAS3 else {"nadam"}


def model(masters, factory, *, build_and_bind=True, trainable_drive=True):
    cell, _, surfaces = build_model(masters, trainable_drive=trainable_drive)
    optimizer = factory()
    trainable = [cell.weight_loss_values(name) for name in surfaces if cell.weight_loss_values(name).trainable]
    if build_and_bind:
        optimizers.build_optimizer(optimizer, trainable)
        cell.bind_optimizer_checkpoint_slots(optimizer)
    return cell, optimizer, surfaces, trainable


def assert_same_state(source, source_optimizer, target, target_optimizer, surfaces, exact=True):
    compare = np.testing.assert_array_equal if exact else (
        lambda a, b: np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-9)
    )
    for name in surfaces:
        compare(target.canonical_weight_values(name), source.canonical_weight_values(name))
        expected = canonical_slots(source, source_optimizer, name)
        actual = canonical_slots(target, target_optimizer, name)
        assert len(actual) == len(expected), name
        for a, b in zip(actual, expected):
            compare(a, b)


def trained_checkpoint(masters, factory, path, trainable_drive=True):
    source, optimizer, surfaces, _ = model(masters, factory, trainable_drive=trainable_drive)
    # Three updates leave every slot nonzero, including Keras 3 gradient
    # accumulators (reset after each applied second step).
    for seed in (1, 2, 3):
        step(source, optimizer, surfaces, seed)
    return source, optimizer, surfaces, checkpoint(source, optimizer).save(str(path))


@pytest.mark.parametrize("name", sorted(FACTORIES))
@pytest.mark.parametrize("source_masters,target_masters", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("order", ["build_bind_restore", "restore_build_bind", "build_restore_bind"])
def test_supported_optimizer_round_trip_in_any_restore_order(
    forward_free, tmp_path, name, source_masters, target_masters, order,
):
    factory = FACTORIES[name]
    drive = not (KERAS3 and order == "restore_build_bind")
    source, source_optimizer, surfaces, path = trained_checkpoint(
        source_masters, factory, tmp_path / "source", trainable_drive=drive,
    )
    for surface in surfaces:
        for slot in canonical_slots(source, source_optimizer, surface):
            assert np.any(slot != 0), (name, surface)

    def resume(masters):
        cell, optimizer, _, trainable = model(
            masters, factory, build_and_bind=order == "build_bind_restore",
            trainable_drive=drive,
        )
        if order == "build_restore_bind":
            optimizers.build_optimizer(optimizer, trainable)
        status = checkpoint(cell, optimizer).restore(path)
        if order == "restore_build_bind":
            optimizers.build_optimizer(optimizer, trainable)
        if order != "build_bind_restore":
            cell.bind_optimizer_checkpoint_slots(optimizer)
        status.assert_consumed()
        return cell, optimizer

    target, target_optimizer = resume(target_masters)
    # Develop-format control: canonical-order masters resumed in the same order.
    control, control_optimizer = resume(False)
    exact = name not in REDUCING
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)
    # The next identical canonical update continues identically.
    for cell, optimizer in ((source, source_optimizer), (target, target_optimizer),
                            (control, control_optimizer)):
        step(cell, optimizer, surfaces, 4)
    assert_same_state(control, control_optimizer, target, target_optimizer, surfaces, exact=exact)
    if name not in UNCHECKPOINTED_STATE:
        assert_same_state(source, source_optimizer, target, target_optimizer, surfaces, exact=exact)
    # Re-saving writes the canonical layout of the source checkpoint.
    second = checkpoint(target, target_optimizer).save(str(tmp_path / "target"))
    assert tf.train.load_checkpoint(second).get_variable_to_shape_map() == (
        tf.train.load_checkpoint(path).get_variable_to_shape_map()
    )


class _RecordingEngine:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.learning_rule = SimpleNamespace(build=lambda rnn: None, uses_bptt=True)
        self.trained = False

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self):
        self.trained = True


def _resume_rnn(cell, trainable, dtype=tf.float32):
    rnn = RNN(cell_params={"hard_reset": cell._hard_reset})
    rnn._model_built = True
    rnn._cell = cell
    rnn.extractor_model = object()
    rnn.strategy = tf.distribute.get_strategy()
    rnn.dtype = dtype
    rnn.model = SimpleNamespace(trainable_variables=trainable)
    return rnn


@pytest.mark.parametrize("source_masters,target_masters", [(False, True), (True, False)])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_rnn_train_resume_restores_before_optimizer_build(
    forward_free, tmp_path, source_masters, target_masters, dtype,
):
    """Public resume flow: build the RNN, restore, then RNN.train builds and binds."""
    factory = FACTORIES["adam"]
    source, source_optimizer, surfaces, path = trained_checkpoint(
        source_masters, factory, tmp_path / "source", trainable_drive=not KERAS3,
    )
    target, target_optimizer, _, trainable = model(
        target_masters, factory, build_and_bind=False, trainable_drive=not KERAS3,
    )
    status = checkpoint(target, target_optimizer).restore(path)
    engine = _RecordingEngine(target_optimizer)
    _resume_rnn(target, trainable, dtype).train(engine)
    assert engine.trained
    if dtype == tf.float16:
        assert isinstance(engine.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
    if dtype == tf.float32 or KERAS3:
        # TF-Keras 2's loss-scale wrapper adds unsaved variables to the root's graph.
        status.assert_consumed()
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)
    step(source, source_optimizer, surfaces, 4)
    step(target, target_optimizer, surfaces, 4)
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)


@pytest.mark.parametrize("source_masters,target_masters", [(False, True), (True, False)])
def test_training_engine_binds_before_its_first_update(
    forward_free, tmp_path, monkeypatch, source_masters, target_masters,
):
    """Direct TrainingEngine use: restore before the optimizer exists, then train_step."""
    factory = FACTORIES["adam"]
    source, source_optimizer, surfaces, path = trained_checkpoint(
        source_masters, factory, tmp_path / "source", trainable_drive=not KERAS3,
    )
    target, target_optimizer, _, trainable = model(
        target_masters, factory, build_and_bind=False, trainable_drive=not KERAS3,
    )
    status = checkpoint(target, target_optimizer).restore(path)
    rnn = _resume_rnn(target, trainable)
    engine = training.TrainingEngine(rnn, n_epochs=1, steps_per_epoch=1)
    engine.set_optimizer(target_optimizer)
    seen = []

    def update(x, y, init_state):
        seen.append(optimizers.optimizer_is_built(engine.optimizer))
        step(target, engine.optimizer, surfaces, 4)
        return {}

    monkeypatch.setattr(engine, "_train_step_unbound", update)
    engine.train_step(None, None, None)
    # CSR masters: built and bound before the update; canonical masters build lazily.
    assert seen == [target_masters]
    status.assert_consumed()
    step(source, source_optimizer, surfaces, 4)
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)


@pytest.mark.parametrize("source_masters,target_masters", [(False, True), (True, False)])
def test_engine_bind_before_restore_resumes_trainable_named_inputs(
    forward_free, tmp_path, source_masters, target_masters,
):
    """Recommended resume flow: bind (builds the optimizer), restore, then train."""
    factory = FACTORIES["adam"]
    source, source_optimizer, surfaces, path = trained_checkpoint(
        source_masters, factory, tmp_path / "source",
    )
    target, target_optimizer, _, trainable = model(target_masters, factory, build_and_bind=False)
    rnn = _resume_rnn(target, trainable)
    engine = training.TrainingEngine(rnn, n_epochs=1, steps_per_epoch=1)
    engine.set_optimizer(target_optimizer)
    assert engine.build_optimizer() is True
    assert optimizers.optimizer_is_built(target_optimizer)
    checkpoint(target, target_optimizer).restore(path).assert_consumed()
    engine_record = _RecordingEngine(target_optimizer)
    rnn.train(engine_record)
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)
    step(source, source_optimizer, surfaces, 4)
    step(target, target_optimizer, surfaces, 4)
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)


def _checkpoint_tensors(path):
    reader = tf.train.load_checkpoint(path)
    return {
        key: reader.get_tensor(key)
        for key in reader.get_variable_to_shape_map()
        if not key.startswith(("save_counter", "_CHECKPOINTABLE_OBJECT_GRAPH"))
    }


def _assert_same_checkpoint(first, second):
    first, second = _checkpoint_tensors(first), _checkpoint_tensors(second)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def _engine(rnn, optimizer, monkeypatch):
    engine = training.TrainingEngine(rnn, n_epochs=1, steps_per_epoch=1)
    engine.set_optimizer(optimizer)
    trained = []
    monkeypatch.setattr(engine, "train", lambda: trained.append(engine.optimizer))
    return engine, trained


@pytest.mark.parametrize("source_masters,target_masters", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_build_optimizer_prepares_the_optimizer_that_train_applies(
    forward_free, tmp_path, monkeypatch, source_masters, target_masters, dtype,
):
    """Mixed-precision checkpoint saved after RNN.train, resumed by
    build -> TrainingEngine.build_optimizer -> restore -> RNN.train."""
    factory = FACTORIES["adam"]
    source, source_inner, surfaces, trainable = model(
        source_masters, factory, build_and_bind=False,
    )
    source_engine, _ = _engine(_resume_rnn(source, trainable, dtype), source_inner, monkeypatch)
    _resume_rnn(source, trainable, dtype).train(source_engine)
    source_optimizer = source_engine.optimizer
    wrapper = tf.keras.mixed_precision.LossScaleOptimizer
    assert isinstance(source_optimizer, wrapper) is (dtype == tf.float16)
    for seed in (1, 2, 3):
        step(source, source_optimizer, surfaces, seed)
    path = checkpoint(source, source_optimizer).save(str(tmp_path / "source"))

    target, target_inner, _, target_trainable = model(
        target_masters, factory, build_and_bind=False,
    )
    rnn = _resume_rnn(target, target_trainable, dtype)
    engine, trained = _engine(rnn, target_inner, monkeypatch)
    assert engine.build_optimizer() is True
    prepared = engine.optimizer
    assert isinstance(prepared, wrapper) is (dtype == tf.float16)
    if dtype == tf.float16:
        assert prepared.inner_optimizer is target_inner
    assert optimizers.optimizer_is_built(prepared)
    checkpoint(target, prepared).restore(path).assert_consumed()
    rnn.train(engine)
    # train() applies the very optimizer that was restored; it is neither
    # re-wrapped nor rebuilt (Keras 3 would reset the restored loss scale).
    assert trained == [prepared] and engine.optimizer is prepared
    assert_same_state(source, source_optimizer, target, prepared, surfaces)
    _assert_same_checkpoint(path, checkpoint(target, prepared).save(str(tmp_path / "restored")))
    step(source, source_optimizer, surfaces, 4)
    step(target, prepared, surfaces, 4)
    assert_same_state(source, source_optimizer, target, prepared, surfaces)
    _assert_same_checkpoint(
        checkpoint(source, source_optimizer).save(str(tmp_path / "source_next")),
        checkpoint(target, prepared).save(str(tmp_path / "target_next")),
    )


def test_prepared_loss_scale_wrapper_is_not_rebuilt_by_train(forward_free, monkeypatch):
    cell, inner, _, trainable = model(True, FACTORIES["adam"], build_and_bind=False)
    rnn = _resume_rnn(cell, trainable, tf.float16)
    engine, _ = _engine(rnn, inner, monkeypatch)
    assert engine.build_optimizer() is True
    prepared = engine.optimizer
    before = [id(variable) for variable in prepared.variables]
    rnn.train(engine)
    assert engine.optimizer is prepared
    assert [id(variable) for variable in prepared.variables] == before


def test_loss_scale_wrapper_around_a_built_optimizer_is_not_reported_built():
    variable = tf.Variable([1.0, 2.0])
    inner = tf.keras.optimizers.Adam(0.01)
    optimizers.build_optimizer(inner, [variable])
    wrapped = optimizers.loss_scale_optimizer_for(inner, tf.float16)
    assert optimizers.loss_scale_optimizer_for(wrapped, tf.float16) is wrapped
    assert optimizers.loss_scale_optimizer_for(inner, tf.float32) is inner
    if getattr(wrapped, "built", None) is False:  # Keras 3 wrappers own their state.
        assert not optimizers.optimizer_is_built(wrapped)
    optimizers.build_optimizer(wrapped, [variable])
    assert optimizers.optimizer_is_built(wrapped)


# Slotless: Keras SGD without momentum (the training config's sgd is "create_sgd").
SLOTLESS = {"sgd": lambda: tf.keras.optimizers.SGD(0.01)}


@pytest.mark.parametrize("name", ["adam", "sgd", "create_sgd"])
def test_restore_before_build_of_trainable_named_inputs(forward_free, tmp_path, name):
    """Keras 3 cannot restore optimizer-only named-input masters before the build:
    CSR masters raise instead of resuming from unrestored weights, also for slotless
    optimizers (SGD without momentum). TF-Keras 2 tracks them in the cell and resumes."""
    factory = SLOTLESS.get(name) or FACTORIES[name]
    source, source_optimizer, surfaces, path = trained_checkpoint(False, factory, tmp_path / "source")
    target, target_optimizer, _, trainable = model(True, factory, build_and_bind=False)
    checkpoint(target, target_optimizer).restore(path)
    if KERAS3:
        with pytest.raises(ValueError, match="before the optimizer was built, but not the master"):
            _resume_rnn(target, trainable).train(_RecordingEngine(target_optimizer))
    else:
        _resume_rnn(target, trainable).train(_RecordingEngine(target_optimizer))
        assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)
        step(source, source_optimizer, surfaces, 4)
        step(target, target_optimizer, surfaces, 4)
        assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)


@pytest.mark.parametrize("name", ["adam", "sgd"])
def test_separate_model_then_optimizer_restores_resume(forward_free, tmp_path, name):
    """Restoring the model, building, then restoring the optimizer is not an
    incomplete restore."""
    factory = SLOTLESS.get(name) or FACTORIES[name]
    source, source_optimizer, surfaces, path = trained_checkpoint(False, factory, tmp_path / "source")
    target, target_optimizer, _, trainable = model(True, factory, build_and_bind=False)
    checkpoint(target, target_optimizer).restore(path)
    optimizers.build_optimizer(target_optimizer, trainable)
    checkpoint(target, target_optimizer).restore(path).assert_consumed()
    target.bind_optimizer_checkpoint_slots(target_optimizer)
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)


def _split_root_checkpoints(tmp_path, factory, source_masters, trainable_drive):
    source, source_optimizer, surfaces, _ = trained_checkpoint(
        source_masters, factory, tmp_path / "combined", trainable_drive=trainable_drive,
    )
    paths = {
        "model": tf.train.Checkpoint(cell=source).save(str(tmp_path / "model")),
        "optimizer": tf.train.Checkpoint(optimizer=source_optimizer).save(str(tmp_path / "optimizer")),
    }
    return source, source_optimizer, surfaces, paths


def _restore_split_roots(target, optimizer, paths, order):
    roots = {"model": tf.train.Checkpoint(cell=target),
             "optimizer": tf.train.Checkpoint(optimizer=optimizer)}
    for root in order.split("_then_"):
        roots[root].restore(paths[root])


@pytest.mark.parametrize("name", ["adam", "sgd"])
@pytest.mark.parametrize("order", ["model_then_optimizer", "optimizer_then_model"])
@pytest.mark.parametrize("source_masters", [False, True])
def test_separate_model_and_optimizer_checkpoint_roots_resume(
    forward_free, tmp_path, name, order, source_masters,
):
    """Successful independent restores from model-only and optimizer-only roots, in
    either order and before the optimizer is built, are not an incomplete restore."""
    factory = SLOTLESS.get(name) or FACTORIES[name]
    # Keras 3 saves trainable named-input masters only under the optimizer's variable
    # list, which a pre-build restore cannot fill (covered by the test below).
    drive = not KERAS3
    source, source_optimizer, surfaces, paths = _split_root_checkpoints(
        tmp_path, factory, source_masters, drive,
    )
    target, target_optimizer, _, trainable = model(
        True, factory, build_and_bind=False, trainable_drive=drive,
    )
    _restore_split_roots(target, target_optimizer, paths, order)
    _resume_rnn(target, trainable).train(_RecordingEngine(target_optimizer))
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)
    step(source, source_optimizer, surfaces, 4)
    step(target, target_optimizer, surfaces, 4)
    assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)


@pytest.mark.parametrize("name", ["adam", "sgd"])
@pytest.mark.parametrize("order", ["model_then_optimizer", "optimizer_then_model"])
def test_separate_roots_still_detect_missing_named_input_masters(
    forward_free, tmp_path, name, order,
):
    factory = SLOTLESS.get(name) or FACTORIES[name]
    source, source_optimizer, surfaces, paths = _split_root_checkpoints(tmp_path, factory, False, True)
    target, target_optimizer, _, trainable = model(True, factory, build_and_bind=False)
    _restore_split_roots(target, target_optimizer, paths, order)
    if KERAS3:
        with pytest.raises(ValueError, match="master 'drive' before the optimizer was built, but not the master"):
            _resume_rnn(target, trainable).train(_RecordingEngine(target_optimizer))
    else:
        # TF-Keras 2 checkpoints named-input masters in the cell: nothing is missing.
        _resume_rnn(target, trainable).train(_RecordingEngine(target_optimizer))
        assert_same_state(source, source_optimizer, target, target_optimizer, surfaces)


def test_update_before_binding_a_restored_slot_raises(forward_free, tmp_path):
    factory = FACTORIES["adam"]
    _, _, surfaces, path = trained_checkpoint(False, factory, tmp_path / "source")
    target, optimizer, _, trainable = model(True, factory, build_and_bind=False)
    checkpoint(target, optimizer).restore(path)
    step(target, optimizer, surfaces, 4)
    with pytest.raises(ValueError, match="restored into a CSR-ordered variable before"):
        target.bind_optimizer_checkpoint_slots(optimizer, after_update=True)


def test_binding_without_restore_never_reorders_live_slots(forward_free):
    cell, optimizer, surfaces, _ = model(True, FACTORIES["adam"], build_and_bind=False)
    trainable = [cell.weight_loss_values(name) for name in surfaces if cell.weight_loss_values(name).trainable]
    optimizers.build_optimizer(optimizer, trainable)
    step(cell, optimizer, surfaces, 4)
    live = {
        name: [slot.numpy().copy() for slot in csr_master.paired_optimizer_slots(optimizer, cell.weight_loss_values(name))]
        for name in surfaces
    }
    cell.bind_optimizer_checkpoint_slots(optimizer, after_update=True)
    cell.bind_optimizer_checkpoint_slots(optimizer)
    for name in surfaces:
        slots = csr_master.paired_optimizer_slots(optimizer, cell.weight_loss_values(name))
        for slot, before in zip(slots, live[name]):
            np.testing.assert_array_equal(slot.numpy(), before)


class _PerEdgeOptimizer(tf.keras.optimizers.Optimizer):
    """Unsupported optimizer with one per-variable state list."""

    def __init__(self, learning_rate, name):
        kwargs = {"name": name}
        if optimizers._optimizer_init_accepts("jit_compile"):
            kwargs["jit_compile"] = False
        try:
            super().__init__(learning_rate=learning_rate, **kwargs)
        except TypeError:
            super().__init__(**kwargs)
            self._learning_rate = self._build_learning_rate(learning_rate)

    def build(self, var_list):
        if getattr(self, "_traces_built", False):
            return
        super().build(var_list)
        self._traces_built = True
        self._traces = []
        for variable in var_list:
            try:
                slot = self.add_variable_from_reference(variable, "trace")
            except TypeError:
                slot = self.add_variable_from_reference(model_variable=variable, variable_name="trace")
            self._traces.append(slot)

    def update_step(self, gradient, variable, learning_rate=None):
        trace = self._traces[self._get_variable_index(variable)]
        trace.assign(0.5 * trace + gradient)
        variable.assign_sub(tf.cast(0.01, variable.dtype) * trace)

    def _get_variable_index(self, variable):
        if hasattr(tf.keras.optimizers.Optimizer, "_get_variable_index"):
            return super()._get_variable_index(variable)
        return self._index_dict[self._var_key(variable)]


class _AdamWithExtraState(tf.keras.optimizers.Adam):
    """Supported class whose subclass adds unpaired per-edge state."""

    def build(self, var_list):
        if getattr(self, "_extra_built", False):
            return
        super().build(var_list)
        self._extra_built = True
        self._extra = []
        for variable in var_list:
            try:
                slot = self.add_variable_from_reference(variable, "extra")
            except TypeError:
                slot = self.add_variable_from_reference(model_variable=variable, variable_name="extra")
            self._extra.append(slot)


@pytest.mark.parametrize("factory,match", [
    (lambda: _PerEdgeOptimizer(learning_rate=0.01, name="per_edge"), "Cannot pair the optimizer state"),
    (lambda: tf.keras.mixed_precision.LossScaleOptimizer(
        _PerEdgeOptimizer(learning_rate=0.01, name="per_edge")), "Cannot pair the optimizer state"),
    (lambda: _AdamWithExtraState(0.01), "not paired with an optimizer variable"),
])
def test_unsupported_optimizer_state_raises_at_binding(forward_free, factory, match):
    with pytest.raises(ValueError, match=match):
        model(True, factory)


def test_unsupported_optimizer_without_csr_masters_is_untouched(forward_free):
    cell, optimizer, _, _ = model(False, lambda: _PerEdgeOptimizer(learning_rate=0.01, name="per_edge"))
    assert cell.bind_optimizer_checkpoint_slots(optimizer) == {}


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_allow_list_pairs_every_per_edge_optimizer_variable(forward_free, name):
    cell, optimizer, surfaces, _ = model(True, FACTORIES[name])
    step(cell, optimizer, surfaces, 5)
    assert csr_master.optimizer_slot_attributes(optimizer) is not None
    paired = {
        id(csr_master._raw_variable(slot))
        for surface in surfaces
        for slot in csr_master.paired_optimizer_slots(optimizer, cell.weight_loss_values(surface))
    }
    shapes = {tuple(cell.weight_loss_values(surface).shape) for surface in surfaces}
    masters = {id(csr_master._raw_variable(cell.weight_loss_values(surface))) for surface in surfaces}
    per_edge = {
        id(csr_master._raw_variable(value))
        for value in csr_master._optimizer_state_variables(optimizer)
        if tuple(value.shape) in shapes
    } - masters
    assert per_edge == paired
