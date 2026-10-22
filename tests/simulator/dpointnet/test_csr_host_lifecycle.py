"""Host-only boundaries and compact native routing for CSR-live masters."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.csr_master import CsrMasterLayout
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops
from bmtk.simulator.dpointnet.custom_ops.master_projection import cpu_connectivity


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def mapping():
    return CsrMasterLayout(cpu_connectivity(
        [[1, 2], [0, 0], [1, 2], [0, 1], [1, 0]], [0, 1, 0, 1, 0],
        3, 2, host_lifecycle=True,
    ), host_lifecycle=True)


def test_compact_host_mapping_and_inverse_not_cached():
    layout = mapping()
    assert layout.csr_to_canonical.dtype == np.uint32
    assert layout.csr_to_canonical.nbytes == 20
    assert not layout.csr_to_canonical.flags.writeable
    assert "_canonical_to_csr" not in vars(layout)
    first, second = layout.canonical_to_csr, layout.canonical_to_csr
    assert first is not second
    np.testing.assert_array_equal(first[layout.csr_to_canonical], np.arange(5))
    canonical = np.arange(15).reshape(5, 3)
    np.testing.assert_array_equal(
        layout.host_to_canonical(layout.host_to_live(canonical)), canonical,
    )
    assert not any(tf.is_tensor(value) for value in vars(layout).values())
    with pytest.raises(ValueError, match="edge dimension"):
        layout.host_to_live(np.arange(6))
    with pytest.raises(ValueError, match="edge dimension"):
        layout.host_to_canonical(np.arange(4))


def test_unused_fixed_four_tables_are_absent_only_in_lifecycle():
    indices = np.column_stack((np.repeat([0, 1], 4), np.tile(np.arange(4), 2)))
    arguments = dict(build_fixed4_incoming=True, build_compact_pairs=True, sort_by_target=True)
    old = ops.build_csr_connectivity(indices, np.zeros(8), 4, 2, 1, **arguments)
    new = ops.build_csr_connectivity(
        indices, np.zeros(8), 4, 2, 1, **arguments, host_lifecycle=True,
    )
    try:
        assert old["fixed4_incoming"] and not new["fixed4_incoming"]
        for conn, has_ids in ((old, True), (new, False)):
            assert conn["device_edge_ids"] is has_ids
            assert (conn["canonical_edge_id_offset"] is not None) is has_ids
            assert len(conn.metadata_resource_handles) == 1
            assert conn.metadata_resource_handles[0] is conn["metadata_handle"]
            spec, = conn["metadata_resource_specs"]
            assert spec["handle"] is conn["metadata_handle"]
            assert spec["dtype"] == conn["metadata_dtype"]
            assert spec["shape"] == conn["metadata_shape"]
            assert spec["payload_bytes"] == conn["metadata_bytes"]
            assert spec["type_logical_min"] == spec["type_logical_max"] == 0
            actual = tf.raw_ops.ReadVariableOp(
                resource=conn["metadata_handle"], dtype=tf.as_dtype(conn["index_dtype"]),
            )
            assert tuple(actual.shape) == conn["metadata_shape"]
            assert int(np.prod(actual.shape)) * actual.dtype.size == conn["metadata_bytes"]
        assert old["metadata_bytes"] - new["metadata_bytes"] == 8 * 4
        assert new["metadata_schema"] == "csr_live_without_edge_ids_v1"
        for name in ("incoming_pre_ids", "incoming_edge_ids", "incoming_types"):
            assert old[name].shape == (8,)
            assert new[name].shape == (0,)
    finally:
        new.close()
        old.close()

@pytest.mark.parametrize("from_live", [False, True])
def test_endpoint_checkpoint_is_canonical_and_portable(tmp_path, from_live):
    from bmtk.simulator.dpointnet.csr_master import canonical_checkpoint
    layout = mapping()
    canonical = np.arange(10).reshape(5, 2)
    source = tf.Variable(layout.host_to_live(canonical) if from_live else canonical)
    target = tf.Variable(canonical if from_live else layout.host_to_live(canonical))
    canonical_checkpoint(source if from_live else target, layout, verify_unchanged=True)
    saved = tf.train.Checkpoint(indices=source).save(str(tmp_path / "endpoints"))
    stored = tf.train.load_checkpoint(saved).get_tensor("indices/.ATTRIBUTES/VARIABLE_VALUE")
    np.testing.assert_array_equal(stored, canonical)
    np.testing.assert_array_equal(source, layout.host_to_live(canonical) if from_live else canonical)
    before = target.numpy().copy()
    tf.train.Checkpoint(indices=target).restore(saved).assert_consumed()
    np.testing.assert_array_equal(target, before)


def test_endpoint_restore_rejects_different_connectivity(tmp_path):
    from bmtk.simulator.dpointnet.csr_master import canonical_checkpoint
    layout = mapping()
    canonical = np.arange(10).reshape(5, 2)
    saved = tf.train.Checkpoint(indices=tf.Variable(canonical[::-1].copy())).save(
        str(tmp_path / "other"))
    target = canonical_checkpoint(tf.Variable(layout.host_to_live(canonical)), layout,
                                  verify_unchanged=True)
    before = target.numpy().copy()
    with pytest.raises(ValueError, match="connectivity differs"):
        tf.train.Checkpoint(indices=target).restore(saved)
    np.testing.assert_array_equal(target, before)


@pytest.mark.parametrize("mode", ["tf_function", "v1_session"])
def test_graph_mode_endpoint_restore_verifies_connectivity(tmp_path, mode):
    from bmtk.simulator.dpointnet.csr_master import canonical_checkpoint
    layout = mapping()
    canonical = np.arange(10).reshape(5, 2)
    other = canonical[::-1].copy()
    live = layout.host_to_live(canonical)
    if mode == "tf_function":
        target = canonical_checkpoint(tf.Variable(live), layout, verify_unchanged=True)
        restore = tf.function(
            lambda value: target._restore_from_tensors({"VARIABLE_VALUE": value})
        )
        with pytest.raises(tf.errors.InvalidArgumentError, match="connectivity differs"):
            restore(tf.constant(other))
        np.testing.assert_array_equal(target, live)
        restore(tf.constant(canonical))
        np.testing.assert_array_equal(target, live)
        return
    paths = {
        name: tf.train.Checkpoint(indices=tf.Variable(values)).save(str(tmp_path / name))
        for name, values in (("other", other), ("same", canonical))
    }
    with tf.Graph().as_default(), tf.compat.v1.Session() as session:
        target = canonical_checkpoint(tf.Variable(live), layout, verify_unchanged=True)
        session.run(target.initializer)
        status = tf.train.Checkpoint(indices=target).restore(paths["other"])
        with pytest.raises(tf.errors.InvalidArgumentError, match="connectivity differs"):
            status.run_restore_ops(session)
        np.testing.assert_array_equal(session.run(target), live)
        tf.train.Checkpoint(indices=target).restore(paths["same"]).run_restore_ops(session)
        np.testing.assert_array_equal(session.run(target), live)


def test_old_direct_to_lifecycle_nonzero_adam_slots_and_export(tmp_path):
    from test_csr_master_layout import fake_cell, raw
    from bmtk.simulator.dpointnet.csr_master import transfer_canonical_state
    source, source_masters = fake_cell(True, False)
    target, target_masters = fake_cell(True, True)
    optimizers = [tf.keras.optimizers.Adam(0.01) for _ in range(2)]
    for optimizer, masters in zip(optimizers, (source_masters, target_masters)):
        optimizer.build(masters[:2])
    def step(optimizer, masters):
        optimizer.apply_gradients([
            (tf.constant([0.1, 0.2, -0.3, 0.4, -0.5]), master) for master in masters[:2]
        ])
    step(optimizers[0], source_masters)
    slots = {
        name: [
            (optimizers[0]._momentums[i], optimizers[1]._momentums[i]),
            (optimizers[0]._velocities[i], optimizers[1]._velocities[i]),
        ] for i, name in enumerate(("<recurrent>", "trainable"))
    }
    slots["frozen"] = []
    transfer_canonical_state(source, target, per_edge_slots=slots, scalar_state=[
        (optimizers[0].iterations, optimizers[1].iterations),
        (optimizers[0].learning_rate, optimizers[1].learning_rate),
    ])
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizers[1], **{f"w_{i}": raw(w) for i, w in enumerate(target_masters)}
    )
    saved = checkpoint.save(str(tmp_path / "converted"))
    step(optimizers[0], source_masters)
    for master in target_masters:
        master.assign(tf.zeros_like(master))
    checkpoint.restore(saved).assert_consumed()
    step(optimizers[1], target_masters)
    for a, b in zip(source_masters, target_masters):
        np.testing.assert_array_equal(a, b)
    for name in slots:
        np.testing.assert_array_equal(
            source.canonical_weight_values(name), target.canonical_weight_values(name),
        )
    for pairs in slots.values():
        for a, b in pairs:
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [tf.int32, tf.int64, tf.uint32])
def test_prepared_loss_indices_are_host_only_and_exact(dtype):
    layout = mapping()
    order = tf.constant([4, 2, 1, 3, 0], dtype)
    actual = layout.canonical_indices_to_live(order)
    assert "CPU:" in actual.device
    np.testing.assert_array_equal(layout.csr_to_canonical[actual.numpy()], order)
    with pytest.raises(ValueError, match="before tracing"):
        tf.function(layout.canonical_indices_to_live)(order)
    with pytest.raises(ValueError, match="outside"):
        layout.canonical_indices_to_live(tf.constant([-1]))


def test_csr_master_option_always_uses_host_lifecycle():
    from test_precision_credit import make_cell
    from test_master_projection import require_cpu_reference
    require_cpu_reference()
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    network, inputs, arguments = make_cell(
        "nest", use_csr_master_weights=True, train_recurrent=True, return_spec=True,
    )
    for props in inputs.values():
        props["input_type"] = "spikes"
    cell = GLIF3Cell(network, inputs, **arguments)
    assert cell._csr_master_weights and cell._csr_host_lifecycle
    assert cell.recurrent_master_layout.host_lifecycle
    assert cell.recurrent_master_layout.csr_to_canonical.dtype == np.uint32


def test_all_runtime_edge_fields_and_masks_follow_exact_stable_order(monkeypatch):
    from test_master_projection import require_cpu_reference
    require_cpu_reference()
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    import bmtk.simulator.dpointnet.csr_master as module
    network, inputs, arguments = make_cell(
        "nest", two_inputs=True, train_recurrent=True, return_spec=True,
    )
    edge_indices = np.array([[1, 1], [0, 0], [1, 1], [1, 0], [0, 1]])
    for props in [network["synapses"], *inputs.values()]:
        props["indices"] = edge_indices.copy()
        props["weights"] = np.array([0.11, -0.23, 0.37, -0.41, 0.59]) * 100
        props["delays"] = np.array([2., 1., 2., 1., 1.])
        props["syn_ids"] = np.zeros(5, np.int64)
    for props in inputs.values():
        props["input_type"] = "spikes"
    # Control: canonical-order masters (use_csr_master_weights=False, the previous path).
    control = GLIF3Cell(network, inputs, **{**arguments, "use_csr_master_weights": False})
    monkeypatch.setattr(module, "_permute", lambda *args: pytest.fail("Initialization used TF permutation"))
    cell = GLIF3Cell(network, inputs, **{**arguments, "use_csr_master_weights": True})
    for name in ("<recurrent>", *inputs):
        layout = cell.recurrent_master_layout if name == "<recurrent>" else cell.inputs[name]["master_layout"]
        actual_indices = cell.recurrent_indices if name == "<recurrent>" else cell.inputs[name]["input_indices"]
        reference_indices = control.recurrent_indices if name == "<recurrent>" else control.inputs[name]["input_indices"]
        actual_types = cell.syn_ids if name == "<recurrent>" else cell.inputs[name]["input_syn_ids"]
        reference_types = control.syn_ids if name == "<recurrent>" else control.inputs[name]["input_syn_ids"]
        actual = cell.weight_loss_values(name)
        reference = control.weight_loss_values(name)
        np.testing.assert_array_equal(actual, layout.host_to_live(reference.numpy()))
        np.testing.assert_array_equal(actual_indices, layout.host_to_live(reference_indices))
        np.testing.assert_array_equal(actual_types, layout.host_to_live(reference_types))
        np.testing.assert_array_equal(
            actual.constraint.condition, layout.host_to_live(np.asarray(reference.constraint.condition)),
        )
        assert "CPU:" in actual_indices.device and "CPU:" in actual_types.device
        np.testing.assert_array_equal(cell.canonical_weight_values(name), reference)
        actual.assign(tf.where(actual.constraint.condition, -tf.ones_like(actual), tf.ones_like(actual)))
        actual.assign(actual.constraint(actual))
        np.testing.assert_array_equal(actual, np.zeros(5))


def test_host_initialization_hash_export_never_calls_tf_gather(monkeypatch):
    from test_master_projection import make_model
    cell, _, _, _ = make_model(monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("Host boundary called tf.gather")
    monkeypatch.setattr(tf, "gather", forbidden)
    for name in ("<recurrent>", *cell.inputs):
        values = cell.canonical_weight_values(name)
        assert isinstance(values, tf.Tensor)
        assert values.dtype == tf.float32
        assert values.shape == cell.weight_loss_values(name).shape
        assert "CPU:" in values.device
        cell.assign_canonical_weights(values, name)
        cell.weight_loss_indices(np.arange(len(values), dtype=np.int64), name)
        cell.weight_loss_reference(values, name)
    with pytest.raises(ValueError, match="host boundary"):
        tf.function(cell.canonical_weight_values)()


@pytest.mark.parametrize("deep_copy", [False, True])
def test_runtime_metadata_and_canonical_export(monkeypatch, deep_copy):
    import pandas as pd
    from test_master_projection import make_model
    from bmtk.simulator.dpointnet.weights import ModelWeights
    cell, _, _, _ = make_model(monkeypatch)
    names = ["<recurrent>", *cell.inputs]
    expected = {}
    for name in names:
        layout = cell.recurrent_master_layout if name == "<recurrent>" else cell.inputs[name]["master_layout"]
        runtime = cell.recurrent_indices if name == "<recurrent>" else cell.inputs[name]["input_indices"]
        assert "CPU:" in runtime.device
        factors = np.arange(int(runtime.shape[0]), dtype=np.float32) + 1.5
        if name == "<recurrent>":
            cell._recurrent_export_factor = layout.host_to_live(factors)
        else:
            cell.inputs[name]["export_factor"] = layout.host_to_live(factors)
        expected[name] = cell.canonical_weight_values(name).numpy() * factors
    rnn = SimpleNamespace(
        inputs_populations=list(cell.inputs),
        get_network=lambda name: SimpleNamespace(
            connection_table=pd.DataFrame({"row": np.arange(2)}), canonical_edge_order=None,
        ),
    )
    conversions = []
    canonical = cell.canonical_weight_values
    monkeypatch.setattr(
        cell, "canonical_weight_values",
        lambda name="<recurrent>": conversions.append(name) or canonical(name),
    )
    export = ModelWeights(rnn, cell, deep_copy=deep_copy)
    if deep_copy:
        assert conversions == names
        assert all(isinstance(export._network_weights[name], np.ndarray) for name in names)
    else:
        # Live exports retain only the CSR-ordered masters, not a canonical copy.
        assert conversions == []
        for name in names:
            assert export._network_weights[name] is cell.weight_loss_values(name)
    for name in names:
        cell.weight_loss_values(name).assign(cell.weight_loss_values(name) * 1.25)
    frame = export.to_dataframe()
    np.testing.assert_array_equal(
        frame["syn_weight"], np.concatenate([expected[name] * (1 if deep_copy else 1.25) for name in names]),
    )


@pytest.mark.parametrize("index_dtype", [tf.uint32, tf.int64])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("dtype,batch,packed,small,event", [
    (np.float16, 1, False, True, False),
    (np.float16, 3, False, False, True),
    (np.float16, 32, True, False, False),
    (np.float16, 32, False, False, True),
    (np.float32, 32, False, False, False),
])
def test_gpu_compact_metadata_native_oracle(dtype, batch, packed, small, event, index_dtype, monkeypatch, typed):
    if not ops.combined_sweep_available():
        pytest.skip("Focused native GPU qualification requires rebuilt operators")
    if (packed or event) and index_dtype != tf.uint32:
        pytest.skip("Packed/event carrier backward intentionally requires uint32 routing")
    from test_master_projection import oracle
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: index_dtype)
    rng = np.random.default_rng(670)
    width, posts, edges = 7, 5, 41
    indices = np.column_stack((rng.integers(posts, size=edges), rng.integers(width, size=edges)))
    types = rng.integers(3, size=edges)
    weights = rng.uniform(-0.7, 0.9, edges).astype(np.float32)
    z = rng.integers(0, 3, (batch, width)).astype(dtype)
    z[:, 0] = 0
    basis = rng.uniform(-0.3, 0.8, (3, 4)).astype(dtype)
    upstream = rng.uniform(-0.4, 0.5, (batch, posts, 4)).astype(dtype)
    args = dict(build_compact_pairs=True, sort_by_target=True)
    full = ops.build_csr_connectivity(indices, types, width, posts, 3, **args)
    compact = ops.build_csr_connectivity(
        indices, types, width, posts, 3, **args, host_lifecycle=True, uint8_types=typed,
    )
    try:
        layout = CsrMasterLayout(compact, host_lifecycle=True)
        expected_bytes = (3 * edges + width + 1 + 2 * compact["n_pairs"]) * index_dtype.size
        if typed:
            expected_bytes -= (index_dtype.size - 1) * edges - (-edges) % index_dtype.size
        assert compact["metadata_bytes"] == expected_bytes
        metadata = tf.raw_ops.ReadVariableOp(resource=compact["metadata_handle"], dtype=index_dtype)
        old = tf.raw_ops.ReadVariableOp(resource=full["metadata_handle"], dtype=index_dtype)
        if not typed:
            np.testing.assert_array_equal(metadata, np.r_[old.numpy()[:2*edges+width+1], old.numpy()[3*edges+width+1:]])
        master = tf.Variable(layout.host_to_live(weights))
        spikes = tf.Variable(z)
        def evaluate():
            with tf.GradientTape() as tape:
                proxy = ops.fused_spike_currents(
                    spikes, master, master, compact, tf.constant(basis), posts, True,
                    vjp_only=True, csr_master_weights=True,
                    use_packed_sm120_backward=packed, use_small_batch_backward=small,
                    write_csr_weight_gradient=True,
                )
                out = ops.fused_combined_currents(
                    [spikes, spikes], [master, master], [compact, compact],
                    tf.constant(basis), [proxy, proxy], csr_master_weights=True,
                    gather_sources=True,
                )
                loss = tf.reduce_sum(tf.cast(out, tf.float32) * upstream.reshape(-1, 4))
            return out, tape.gradient(loss, (spikes, master))
        out, (dz, dw) = tf.function(evaluate)()
        expected = oracle(indices, types, weights, z, basis, upstream)
        tol = 0.02 if dtype == np.float16 else 2e-5
        np.testing.assert_allclose(out, 2 * expected[0], rtol=tol, atol=tol)
        np.testing.assert_allclose(dz, 2 * expected[1], rtol=tol, atol=tol)
        np.testing.assert_allclose(dw, layout.host_to_live(2 * expected[2]), rtol=tol, atol=tol)
        kw = dict(Tindex=index_dtype, n_post=posts, n_edges=edges, n_pairs=compact["n_pairs"])
        kw.update(ops._uint8_type_options(compact))
        if event:
            event_dz, event_dw = ops._OPS.dpointnet_csr_spike_grad_accumulate_csr_master(
                tf.constant(z), tf.constant(upstream.reshape(-1, 4)), compact["metadata_handle"],
                master, tf.constant(basis), tf.constant(1., tf.as_dtype(dtype)),
                tf.fill((edges,), 0.125), use_javier_batch32_backward=True, **kw,
            )
            np.testing.assert_allclose(event_dz, expected[1], rtol=tol, atol=tol)
            np.testing.assert_allclose(
                event_dw, layout.host_to_live(expected[2]) + 0.125, rtol=tol, atol=tol,
            )
        named = ops._OPS.dpointnet_csr_weight_grad(
            tf.constant(z), tf.constant(upstream.reshape(-1, 4)), compact["metadata_handle"],
            tf.constant(basis), write_csr_weight_gradient=True, use_packed_sm120_backward=packed, **kw,
        )
        np.testing.assert_allclose(named, layout.host_to_live(expected[2]), rtol=2e-5, atol=2e-5)
        with pytest.raises(tf.errors.InvalidArgumentError, match="metadata|direct CSR host lifecycle"):
            ops._OPS.dpointnet_csr_weight_grad(
                tf.constant(z), tf.constant(upstream.reshape(-1, 4)), compact["metadata_handle"],
                tf.constant(basis), write_csr_weight_gradient=False, **kw,
            )
        with pytest.raises(ValueError if typed else tf.errors.InvalidArgumentError, match="metadata|canonical edge IDs"):
            ops.reorder_csr_values(master, compact)
    finally:
        compact.close()
        full.close()
