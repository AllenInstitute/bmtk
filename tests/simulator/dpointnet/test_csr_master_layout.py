"""CSR ownership, canonical migration, and fail-closed checkpoint boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.csr_master import (
    CsrMasterLayout, guard_csr_checkpoint, transfer_canonical_state,
)
from bmtk.simulator.dpointnet.custom_ops.master_projection import cpu_connectivity


def layout(host_lifecycle=False):
    # Duplicate endpoints remain distinct and preserve their initial row ordering.
    indices = np.array([[1, 2], [0, 0], [1, 2], [0, 1], [1, 0]])
    return CsrMasterLayout(
        cpu_connectivity(indices, [0, 1, 0, 1, 0], 3, 2, host_lifecycle=host_lifecycle),
        host_lifecycle=host_lifecycle,
    )


def raw(variable):
    value = variable.value
    return variable if callable(value) else value


def test_duplicate_edges_and_reversible_initial_row_order():
    mapping = layout()
    np.testing.assert_array_equal(mapping.csr_to_canonical, [1, 4, 3, 0, 2])
    canonical = tf.constant([0.11, -0.23, 0.37, -0.41, 0.59])
    master = tf.Variable(mapping.to_live(canonical))
    np.testing.assert_array_equal(mapping.to_canonical(master), canonical)
    mapping.assign_canonical(master, canonical * 1.2)
    np.testing.assert_array_equal(master, tf.gather(canonical * 1.2, [1, 4, 3, 0, 2]))
    state = dict(master=canonical, momentum=canonical * 2, velocity=canonical ** 2,
                 constraint=canonical >= 0)
    restored = mapping.convert_state(mapping.convert_state(state, to_live=True), to_live=False)
    for key in state:
        np.testing.assert_array_equal(state[key], restored[key])


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64, tf.uint32])
def test_loss_index_composition_matches_canonical_gather_without_weight_copy(dtype):
    mapping = layout()
    canonical = tf.constant([0.11, -0.23, 0.37, -0.41, 0.59])
    master = tf.Variable(mapping.to_live(canonical))
    group_order = tf.constant([0, 2, 1, 4, 3], dtype=dtype)
    live_group_order = mapping.canonical_indices_to_live(group_order)
    assert live_group_order.dtype == dtype
    np.testing.assert_array_equal(
        tf.gather(master, tf.cast(live_group_order, tf.int64)),
        tf.gather(canonical, tf.cast(group_order, tf.int64)),
    )


@pytest.mark.parametrize("checkpointing", [False, True])
def test_training_engine_never_canonical_restores_live_csr_gradients(checkpointing):
    from bmtk.simulator.dpointnet.training import TrainingEngine, TrainingParameters

    def forbidden_restore(*args, **kwargs):
        pytest.fail("TrainingEngine must not canonical-restore CSR-live master gradients")

    cell = SimpleNamespace(
        _csr_master_weights=True, _use_direct_csr_recurrent_gradient=True,
        restore_segmented_variable_gradients=forbidden_restore,
    )
    core = SimpleNamespace(
        layers=[], output=[
            tf.TensorSpec((None, 7, 1), tf.float32),
            tf.TensorSpec((None, 7, 1), tf.float32),
            tf.TensorSpec((None, 1), tf.float32),
        ],
    )
    engine = TrainingEngine(
        SimpleNamespace(cell=cell, extractor_model=core), 1, 1,
        training_approach="single", gradient_checkpointing=checkpointing,
        gradient_checkpoint_chunk_size=3,
    )
    engine._parameters = [TrainingParameters("test", batch_size=1, seq_len=7)]
    engine.prepare_gradient_checkpointing()
    transform = engine._extractor_forward.variable_gradient_transform
    if checkpointing:
        assert transform is None
    else:
        gradients = (tf.constant([0.13, -0.27, 0.41]),)
        assert transform((), gradients) is gradients


@pytest.mark.parametrize("source_csr", [False, True])
def test_checkpoint_formats_reject_cross_layout_restore(tmp_path, source_csr):
    source = tf.Variable([0.11, -0.23, 0.37])
    target = tf.Variable([1., 2., 3.])
    guard_csr_checkpoint(source if source_csr else target)
    path = tf.train.Checkpoint(master=source).save(str(tmp_path / "layout"))
    before = target.numpy().copy()
    with pytest.raises(tf.errors.NotFoundError):
        tf.train.Checkpoint(master=target).restore(path).assert_consumed()
    np.testing.assert_array_equal(target, before)


def test_checkpoint_rejects_changed_csr_permutation(tmp_path):
    first = layout()
    second = CsrMasterLayout(cpu_connectivity(
        np.array([[0, 0], [0, 1], [1, 2], [1, 2], [1, 0]]),
        [0, 1, 0, 0, 1], 3, 2,
    ))
    source, target = tf.Variable(tf.range(5, dtype=tf.float32)), tf.Variable(tf.zeros(5))
    guard_csr_checkpoint(source, first)
    guard_csr_checkpoint(target, second)
    path = tf.train.Checkpoint(master=source).save(str(tmp_path / "mapping"))
    with pytest.raises(tf.errors.NotFoundError):
        tf.train.Checkpoint(master=target).restore(path).assert_consumed()
    np.testing.assert_array_equal(target, np.zeros(5))


def fake_cell(csr, host_lifecycle=False):
    mapping = layout(host_lifecycle=host_lifecycle)
    canonical = tf.constant([0.11, -0.23, 0.37, -0.41, 0.59])
    masters = [
        tf.Variable(mapping.to_live(canonical) if csr else canonical,
                    trainable=i < 2, name=f"master_{i}")
        for i in range(3)
    ]
    if csr:
        for master in masters:
            guard_csr_checkpoint(master)
    cell = SimpleNamespace(
        _csr_master_weights=csr, recurrent_master_layout=mapping,
        recurrent_weight_values=masters[0],
        inputs={
            "trainable": dict(input_weight_values=masters[1], master_layout=mapping),
            "frozen": dict(input_weight_values=masters[2], master_layout=mapping),
        },
        refresh_recurrent_weight_shadow=lambda: None,
    )

    def canonical_value(name="<recurrent>"):
        master = masters[0] if name == "<recurrent>" else cell.inputs[name]["input_weight_values"]
        return mapping.to_canonical(master) if csr else tf.identity(master)

    cell.canonical_weight_values = canonical_value
    return cell, masters


@pytest.mark.parametrize("source_csr", [False, True])
@pytest.mark.parametrize("host_lifecycle", [False, True])
def test_explicit_old_new_checkpoint_adam_migration_and_resumed_update(tmp_path, source_csr, host_lifecycle):
    source, source_masters = fake_cell(source_csr, host_lifecycle)
    target, target_masters = fake_cell(not source_csr, host_lifecycle)
    source_opt, target_opt = tf.keras.optimizers.Adam(0.01), tf.keras.optimizers.Adam(0.01)
    for opt, masters in ((source_opt, source_masters), (target_opt, target_masters)):
        opt.build(masters[:2])
    canonical_gradient = tf.constant([0.13, -0.7, 0.21, 0.33, -0.19])

    def step(cell, opt, masters):
        gradient = (
            cell.recurrent_master_layout.to_live(canonical_gradient)
            if cell._csr_master_weights else canonical_gradient
        )
        opt.apply_gradients([(gradient, master) for master in masters[:2]])

    step(source, source_opt, source_masters)
    source_ckpt = tf.train.Checkpoint(
        optimizer=source_opt, **{f"master_{i}": raw(w) for i, w in enumerate(source_masters)}
    )
    saved = source_ckpt.save(str(tmp_path / "source"))
    values = [w.numpy().copy() for w in source_masters]
    for master in source_masters:
        master.assign(tf.zeros_like(master))
    source_ckpt.restore(saved).assert_consumed()
    for master, expected in zip(source_masters, values):
        np.testing.assert_array_equal(master, expected)
    # Adam exposes exactly two per-edge slots per trainable master.
    slots = {}
    for i, name in enumerate(("<recurrent>", "trainable")):
        slots[name] = [
            (source_opt._momentums[i], target_opt._momentums[i]),
            (source_opt._velocities[i], target_opt._velocities[i]),
        ]
    slots["frozen"] = []
    transfer_canonical_state(source, target, per_edge_slots=slots, scalar_state=[
        (source_opt.iterations, target_opt.iterations),
        (source_opt.learning_rate, target_opt.learning_rate),
    ])
    for name in slots:
        np.testing.assert_array_equal(source.canonical_weight_values(name),
                                      target.canonical_weight_values(name))
    target_ckpt = tf.train.Checkpoint(
        optimizer=target_opt, **{f"master_{i}": raw(w) for i, w in enumerate(target_masters)}
    )
    saved_target = target_ckpt.save(str(tmp_path / "converted"))
    for master in target_masters:
        master.assign(tf.zeros_like(master))
    target_ckpt.restore(saved_target).assert_consumed()
    step(source, source_opt, source_masters)
    step(target, target_opt, target_masters)
    for name in slots:
        np.testing.assert_array_equal(source.canonical_weight_values(name),
                                      target.canonical_weight_values(name))
    assert int(source_opt.iterations.numpy()) == int(target_opt.iterations.numpy()) == 2


def test_native_reader_is_raw_contiguous_float_pointer():
    from pathlib import Path
    import bmtk.simulator.dpointnet.custom_ops.csr_spike_ops as ops
    source = Path(ops.__file__).with_name("csr_spike_ops.cu.cc").read_text()
    assert "std::conditional_t<Canonical, const float*, const T*>" in source
    assert "struct CanonicalWeights" not in source
    assert "return values.flat<float>().data();" in source


def test_keras_tracked_master_uses_guarded_checkpoint_key(tmp_path):
    layer = tf.keras.layers.Layer()
    master = layer.add_weight(
        name="csr_weight", shape=(3,), dtype=tf.float32,
        initializer=tf.keras.initializers.Constant([0.17, -0.31, 0.43]),
    )
    guard_csr_checkpoint(master)
    checkpoint = tf.train.Checkpoint(layer=layer, master=raw(master))
    path = checkpoint.save(str(tmp_path / "keras"))
    keys = [name for name, _ in tf.train.list_variables(path)]
    assert any(name.endswith("/CSR_MASTER_V1") for name in keys)
    master.assign([1., 2., 3.])
    checkpoint.restore(path).assert_consumed()
    np.testing.assert_array_equal(master, np.array([0.17, -0.31, 0.43], np.float32))


def test_one_replica_mirrored_master_checkpoint_guard(tmp_path):
    strategy = tf.distribute.MirroredStrategy(devices=["/CPU:0"])
    with strategy.scope():
        layer = tf.keras.layers.Layer()
        master = layer.add_weight(
            name="csr_weight", shape=(3,), dtype=tf.float32,
            initializer=tf.keras.initializers.Constant([0.17, -0.31, 0.43]),
        )
        mapping = CsrMasterLayout(cpu_connectivity(
            [[1, 1], [0, 0], [1, 1]], [0, 1, 0], 2, 2,
        ))
        guard_csr_checkpoint(master, mapping)
    checkpoint = tf.train.Checkpoint(master=raw(master))
    path = checkpoint.save(str(tmp_path / "one_replica"))
    keys = [name for name, _ in tf.train.list_variables(path)]
    assert any("/CSR_MASTER_V1_" in name for name in keys)
    master.assign([1., 2., 3.])
    checkpoint.restore(path).assert_consumed()
    np.testing.assert_array_equal(master, np.array([0.17, -0.31, 0.43], np.float32))


@pytest.mark.parametrize("deep_copy", [False, True])
def test_model_weights_exports_canonical_rows_and_live_assignments(deep_copy):
    import pandas as pd
    from bmtk.simulator.dpointnet.weights import ModelWeights
    cell, masters = fake_cell(True)
    factors = np.array([1., 2., 3., 4., 5.], np.float32)
    cell._recurrent_export_factor = factors
    for net in cell.inputs.values():
        net["export_factor"] = factors
    adapter = SimpleNamespace(
        connection_table=pd.DataFrame({"row": np.arange(5)}),
        canonical_edge_order=None,
    )
    rnn = SimpleNamespace(inputs_populations=list(cell.inputs), get_network=lambda name: adapter)
    exported = ModelWeights(rnn, cell, deep_copy=deep_copy)
    initial = cell.canonical_weight_values().numpy()
    for master in masters:
        master.assign(master * 1.25)
    frame = exported.to_dataframe()
    expected = initial * (1 if deep_copy else 1.25) * factors
    for offset in (0, 5, 10):
        np.testing.assert_array_equal(frame["syn_weight"].to_numpy()[offset:offset + 5], expected)


def test_emd_groups_and_live_gradient_follow_csr_layout(monkeypatch):
    from bmtk.simulator.dpointnet.loss_functions import weight_regularization as wr
    source, source_masters = fake_cell(False)
    target, target_masters = fake_cell(True)
    groups = np.array([0, 1, 0, 2, 1], np.int64)
    monkeypatch.setattr(wr, "_connection_type_ids", lambda *args, **kwargs: groups)
    regularizers = [
        wr.EMDWeightRegularization(
            SimpleNamespace(cell=cell, recurrent_network={}), cost=1.0,
        ) for cell in (source, target)
    ]
    current = tf.constant([0.18, -0.11, 0.27, -0.47, 0.63])
    source_masters[0].assign(current)
    target_masters[0].assign(target.recurrent_master_layout.to_live(current))

    @tf.function
    def evaluate():
        with tf.GradientTape(persistent=True) as tape:
            values = [regularizer() for regularizer in regularizers]
        return values, [
            tape.gradient(value, master) for value, master in zip(
                values, (source_masters[0], target_masters[0])
            )
        ]

    values, gradients = evaluate()
    np.testing.assert_array_equal(values[0], values[1])
    np.testing.assert_array_equal(
        gradients[0], target.recurrent_master_layout.to_canonical(gradients[1])
    )
