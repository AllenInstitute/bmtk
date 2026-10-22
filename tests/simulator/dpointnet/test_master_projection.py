"""Private direct-master native CPU/GPU correctness and lifecycle tests."""

import os

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops.master_projection import (
    cpu_connectivity, project,
)


DEVICE = os.environ.get("_MASTER_TEST_DEVICE", "/CPU:0")


@pytest.fixture(autouse=True)
def require_native_library():
    from bmtk.simulator.dpointnet.custom_ops.master_projection import reference_available
    if not reference_available():
        pytest.skip("Built native operators (and their CPU reference kernels) cannot be loaded here.")


@pytest.fixture(autouse=True)
def policy():
    old = tf.keras.mixed_precision.global_policy()
    with tf.device(DEVICE):
        yield
    tf.keras.mixed_precision.set_global_policy(old)


def fixture(dtype, index_dtype, batch=3, empty=False):
    rng = np.random.default_rng(431)
    width, posts, edges = 7, 5, 39 if not empty else 0
    indices = np.column_stack((
        rng.integers(posts, size=edges), rng.integers(width, size=edges),
    ))
    types = rng.integers(3, size=edges)
    master = rng.uniform(-1.5, 1.5, edges).astype(np.float32)
    z = rng.integers(0, 4, (batch, width)).astype(dtype)
    z[:, 0] = 0
    basis = rng.uniform(-0.8, 1.2, (3, 4)).astype(dtype)
    upstream = rng.uniform(-0.7, 0.7, (batch, posts, 4)).astype(np.float32)
    conn = cpu_connectivity(indices, types, width, posts)
    metadata = tf.cast(conn["master_metadata"], index_dtype)
    return indices, types, master, z, basis, upstream, metadata


def oracle(indices, types, weights, z, basis, upstream):
    batch, posts, _ = upstream.shape
    output = np.zeros_like(upstream, dtype=np.float64)
    dz = np.zeros_like(z, dtype=np.float64)
    dw = np.zeros_like(weights, dtype=np.float64)
    for edge, (post, pre) in enumerate(indices):
        for sample in range(batch):
            credit = 0.0
            for k in range(4):
                coefficient = float(basis[types[edge], k])
                output[sample, post, k] += float(z[sample, pre]) * float(weights[edge]) * coefficient
                credit += float(upstream[sample, post, k]) * coefficient
            dz[sample, pre] += float(weights[edge]) * credit
            dw[edge] += float(z[sample, pre]) * credit
    return output.reshape(batch * posts, 4), dz, dw


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("index_dtype", [tf.uint32, tf.int64])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("batch", [1, 3, 32])
@pytest.mark.parametrize("empty", [False, True])
def test_independent_fp64_forward_and_vjps(dtype, index_dtype, compiled, batch, empty):
    indices, types, w, z, basis, dy, metadata = fixture(dtype, index_dtype, batch, empty)
    zs, ws = tf.Variable(z), tf.Variable(w)

    def evaluate():
        with tf.GradientTape() as tape:
            out = project(zs, ws, metadata, tf.constant(basis), 5, len(w))
            loss = tf.reduce_sum(out * dy.reshape(-1, 4))
        return out, *tape.gradient(loss, (zs, ws))

    actual = tf.function(evaluate)() if compiled else evaluate()
    expected = oracle(indices, types, w, z, basis, dy)
    for i, (got, reference) in enumerate(zip(actual, expected)):
        tolerance = 1e-3 if i == 1 and dtype == np.float16 else 6e-6
        np.testing.assert_allclose(got, reference, atol=tolerance, rtol=tolerance)
    assert actual[0].dtype == tf.float32 and actual[2].dtype == tf.float32
    if not empty:
        assert np.any(actual[1].numpy()[:, 0] != 0), "Silent source credit was lost"


def test_master_precision_is_not_a_reexpanded_half_shadow():
    indices = np.array([[0, 0], [0, 1]])
    conn = cpu_connectivity(indices, np.zeros(2, np.int64), 2, 1)
    master = tf.Variable([1.0003, 0.00003], dtype=tf.float32)
    result = project(
        tf.ones((1, 2), tf.float16), master, conn["master_metadata"],
        tf.ones((1, 4), tf.float16), 1, 2,
    )
    expected = np.float32(1.0003) + np.float32(0.00003)
    np.testing.assert_array_equal(result, np.full((1, 4), expected, np.float32))
    assert result.numpy()[0, 0] != np.float32(
        np.float16(master.numpy()[0]) + np.float16(master.numpy()[1])
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_custom_gradient_variable_boundary_assignment_update_restore(tmp_path, compiled):
    indices, types, w, z, basis, dy, metadata = fixture(np.float16, tf.int64)
    master = tf.Variable(w, name="canonical_master")
    optimizer = tf.keras.optimizers.Adam(0.01)
    optimizer.build([master])
    checkpoint = tf.train.Checkpoint(master=master, optimizer=optimizer)

    def evaluate(z):
        with tf.GradientTape() as tape:
            tape.watch(z)
            result = project(z, master, metadata, tf.constant(basis), 5, len(w))
            loss = tf.reduce_sum(result * dy.reshape(-1, 4))
        return result, *tape.gradient(loss, (z, master))

    run = tf.function(evaluate) if compiled else evaluate
    first = run(tf.constant(z))
    saved = checkpoint.save(str(tmp_path / "master"))
    master.assign(w + 0.21)
    assigned = run(tf.constant(z))
    reference = oracle(indices, types, master.numpy(), z, basis, dy)
    np.testing.assert_allclose(assigned[0], reference[0], rtol=6e-6, atol=6e-6)
    assert not np.array_equal(first[0], assigned[0])
    optimizer.apply_gradients([(assigned[2], master)])
    updated = run(tf.constant(z))
    assert not np.array_equal(assigned[0], updated[0])
    checkpoint.restore(saved).assert_consumed()
    restored = run(tf.constant(z))
    for a, b in zip(first, restored):
        np.testing.assert_array_equal(a, b)
    assert int(optimizer.iterations.numpy()) == 0
    if compiled:
        functions = run.get_concrete_function(tf.constant(z)).graph.as_graph_def()
        forbidden = ("DpointnetCsrReorder", "DpointnetCsrRestore")
        assert not any(node.op in forbidden for node in functions.node)
        native = [node for node in functions.node if node.op == "DpointnetMasterProjection"]
        assert len(native) == 1
        assert tf.dtypes.as_dtype(native[0].attr["T"].type) == tf.float16


def require_cpu_reference():
    from bmtk.simulator.dpointnet.custom_ops.master_projection import reference_available
    if tf.config.get_visible_devices("GPU"):
        pytest.skip("The CPU CSR-master reference requires no visible GPU (CUDA-hidden suite).")
    if not reference_available():
        pytest.skip("Built native operators (CPU reference kernels) cannot be loaded here.")


def make_model(monkeypatch, enabled=True, mixed=True):
    if enabled:
        require_cpu_reference()
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    network, inputs, arguments = make_cell(
        "nest", two_inputs=True, return_spec=True, train_recurrent=True,
        selective=mixed, use_csr_master_weights=enabled,
    )
    for props in inputs.values():
        props["input_type"] = "spikes"
    cell = GLIF3Cell(network, inputs, **arguments)
    state = list(cell.zero_state(2, cell.compute_dtype))
    state[1] = tf.constant([[1.1, 0.8], [0.4, 1.2]], state[1].dtype)
    width = int(cell.inputs_idx[-1])
    sequence = tf.keras.Input(shape=(None, width), dtype=cell.compute_dtype)
    states = [tf.keras.Input(shape=x.shape[1:], dtype=x.dtype) for x in state]
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    layer._autocast = False
    out = layer(sequence, initial_state=states)
    core = tf.keras.Model([sequence, *states], [*out[0], *out[1:]])
    return cell, core, state, tf.ones((2, 7, width), cell.compute_dtype)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("chunk", [2, 3, 7])
def test_named_input_segmented_remainder_and_master_lifecycle(monkeypatch, tmp_path, compiled, chunk):
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner
    cell, core, state, sequence = make_model(monkeypatch)
    runner = SegmentedRecomputeRunner(core, 7, chunk, 2, differentiate_inputs=True)
    masters = [cell.recurrent_weight_values] + [
        net["input_weight_values"] for net in cell.inputs.values()
    ]
    assert cell.recurrent_weight_values_compute is masters[0]
    assert cell.recurrent_csr_weight_values_compute is None
    for master, net in zip(masters[1:], cell.inputs.values()):
        assert net["input_weight_values_compute"] is master
        assert net["csr_weight_values_compute"] is None
    assert all(master.dtype == "float32" for master in masters)
    assert set(map(id, masters)).issubset(set(map(id, core.trainable_variables)))

    def evaluate(segmented):
        with tf.GradientTape() as tape:
            tape.watch(sequence)
            values = runner(sequence, state) if segmented else core([sequence, *state])
            loss = tf.reduce_sum(tf.cast(values[0], tf.float32)) + tf.reduce_sum(values[1])
        return values, tape.gradient(loss, [sequence, *masters])

    full = tf.function(lambda: evaluate(False)) if compiled else lambda: evaluate(False)
    segment = tf.function(lambda: evaluate(True)) if compiled else lambda: evaluate(True)

    def check():
        a, b = full(), segment()
        for got, expected in zip(tf.nest.flatten(b), tf.nest.flatten(a)):
            assert got is not None and expected is not None
            np.testing.assert_allclose(got, expected, rtol=3e-3, atol=2e-3)
        for gradient in a[1][1:]:
            assert gradient.dtype == tf.float32
            assert np.any(gradient.numpy() != 0)
        return a

    first = check()
    canonical_gradients = first[1][1:]
    assert cell.restore_segmented_variable_gradients(masters, canonical_gradients) is canonical_gradients
    optimizer = tf.keras.optimizers.Adam(0.001)
    optimizer.build(masters)
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, **{f"master_{i}": getattr(w, "value", w) for i, w in enumerate(masters)}
    )
    saved = checkpoint.save(str(tmp_path / "all_masters"))
    optimizer.apply_gradients(zip(first[1][1:], masters))
    for master in masters:
        assert np.all(np.where(
            master.constraint.condition.numpy(), master.numpy() >= 0, master.numpy() <= 0
        ))
    cell.refresh_recurrent_weight_shadow()
    check()
    for master in masters:
        master.assign(master * 1.2)
    assigned = check()
    assert not np.array_equal(first[0][1], assigned[0][1])
    checkpoint.restore(saved).assert_consumed()
    restored = check()
    for got, expected in zip(tf.nest.flatten(restored), tf.nest.flatten(first)):
        np.testing.assert_array_equal(got, expected)


def test_disabled_baseline_keeps_original_weight_shadows(monkeypatch):
    cell, _, _, _ = make_model(monkeypatch, enabled=False)
    assert cell.recurrent_weight_values_compute is not cell.recurrent_weight_values
    assert cell.recurrent_weight_values_compute.dtype == tf.float16
    for net in cell.inputs.values():
        assert net["input_weight_values_compute"] is not net["input_weight_values"]
        assert net["input_weight_values_compute"].dtype == tf.float16


def test_invalid_master_dtype_is_explicit():
    with pytest.raises(ValueError, match="FP32 weights"):
        project(tf.ones((1, 1), tf.float16), tf.ones(1, tf.float16),
                tf.constant([0, 0, 0, 1, 0], tf.int64), tf.ones((1, 4), tf.float16), 1, 1)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_resource_metadata_and_live_silent_credit_scale(dtype):
    from bmtk.simulator.dpointnet.custom_ops.master_projection import connected_project
    indices, types, w, z, basis, dy, metadata = fixture(dtype, tf.int64)
    resource = tf.Variable(metadata, trainable=False)
    conn = {
        "metadata_handle": resource.handle, "index_dtype": "int64",
        "n_post": 5, "n_edges": len(w),
    }
    master = tf.Variable(w)

    @tf.function
    def calculate(z):
        with tf.GradientTape() as tape:
            tape.watch(z)
            out = connected_project(z, master, conn, tf.constant(basis),
                                    tf.constant(0.3, tf.as_dtype(dtype)))
            loss = tf.reduce_sum(out * dy.reshape(-1, 4))
        return out, *tape.gradient(loss, (z, master))

    actual = calculate(tf.constant(z))
    expected = list(oracle(indices, types, w, z, basis, dy))
    expected[1] *= float(np.asarray(0.3, dtype=dtype))
    for i, (got, ref) in enumerate(zip(actual, expected)):
        tolerance = 1e-3 if i == 1 and dtype == np.float16 else 6e-6
        np.testing.assert_allclose(got, ref, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("mixed", [False, True])
def test_internal_poisson_and_frozen_named_masters(monkeypatch, mixed):
    require_cpu_reference()
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    network, inputs, arguments = make_cell(
        "nest", noise=True, two_inputs=True, selective=mixed, return_spec=True,
        train_recurrent=True, use_csr_master_weights=True,
    )
    inputs["background"]["options"]["trainable"] = False
    cell = GLIF3Cell(network, inputs, **arguments)
    net = cell.inputs["background"]
    assert not net["input_weight_values"].trainable
    assert net["input_weight_values_compute"] is net["input_weight_values"]
    assert net["csr_weight_values_compute"] is None
    counts = cell.sample_noise_spikes(100000, tf.constant([0], tf.int64), net).numpy()
    assert counts.dtype == np.int32
    assert abs(counts.mean() - 0.5) < 0.01
    assert abs(counts.var() - 0.5) < 0.015
    assert np.any(counts > 1)
    checkpoint_inputs = tf.zeros((128, 0), cell.compute_dtype)
    state = list(cell.zero_state(128, cell.compute_dtype))

    @tf.function
    def projection():
        return cell._project_step_currents(checkpoint_inputs, state)[0]

    before = projection()
    net["input_weight_values"].assign(net["input_weight_values"] * 1.75)
    after = projection()
    assert not np.array_equal(before, after)


@pytest.mark.parametrize("mixed", [False, True])
def test_cell_recurrent_and_each_named_input_against_fp64(monkeypatch, mixed):
    cell, _, state, _ = make_model(monkeypatch, mixed=mixed)
    dtype = tf.as_dtype(cell.compute_dtype)
    rng = np.random.default_rng(91)
    z = tf.Variable(rng.integers(0, 4, state[0].shape).astype(dtype.as_numpy_dtype))
    z.assign(tf.concat([tf.zeros_like(z[:, :1]), z[:, 1:]], axis=1))
    state[0] = z
    state[7] = tf.constant(rng.integers(0, 4, state[7].shape), dtype=dtype)
    x = tf.constant([[1., 2., 3., 1.], [0., 3., 1., 2.]], dtype=dtype)
    upstream = rng.uniform(-0.5, 0.5, (2, 2, 4)).astype(dtype.as_numpy_dtype)
    basis = cell.synaptic_basis_weights.numpy()
    masters = [cell.recurrent_weight_values] + [
        net["input_weight_values"] for net in cell.inputs.values()
    ]

    @tf.function
    def evaluate():
        with tf.GradientTape() as tape:
            result, _ = cell._project_step_currents(x, state)
            loss = tf.reduce_sum(tf.cast(result, tf.float32) * upstream.reshape(2, -1))
        return result, tape.gradient(loss, [z, *masters])

    result, gradients = evaluate()
    # Endpoint indices and types are stored in live CSR order; the oracle is canonical.
    canonical = cell.recurrent_master_layout.host_to_canonical
    rec = oracle(
        canonical(cell.recurrent_indices.numpy()), canonical(cell.syn_ids.numpy()),
        cell.canonical_weight_values().numpy(), z.numpy(), basis, upstream.astype(np.float32),
    )
    expected = [rec]
    for idx, (name, net) in enumerate(cell.inputs.items()):
        current = x[:, cell.inputs_idx[idx]:cell.inputs_idx[idx + 1]].numpy()
        prior = state[7][:, net["history_start"]:net["history_stop"]].numpy()
        delayed = np.concatenate([current, prior], axis=1)
        to_canonical = net["master_layout"].host_to_canonical
        expected.append(oracle(
            to_canonical(net["input_indices"].numpy()), to_canonical(net["input_syn_ids"].numpy()),
            cell.canonical_weight_values(name).numpy(), delayed, basis, upstream.astype(np.float32),
        ))
    reference = sum(item[0] for item in expected)
    np.testing.assert_allclose(result, reference.reshape(2, -1), rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(gradients[0], rec[1], rtol=1e-3, atol=1e-3)
    layouts = [cell.recurrent_master_layout] + [
        net["master_layout"] for net in cell.inputs.values()
    ]
    for got, reference, layout in zip(gradients[1:], [item[2] for item in expected], layouts):
        reference = layout.to_live(reference)
        np.testing.assert_allclose(got, reference, rtol=6e-6, atol=6e-6)
    assert np.any(gradients[0].numpy()[:, 0] != 0)


# One edge from source 0 to target 0 with type 0: [posts | types | splits | canonical IDs].
VALID_ONE_EDGE = (0, 0, 0, 1, 0)
INVALID_ONE_EDGE = {
    "negative_post": (-1, 0, 0, 1, 0),
    "post_beyond_targets": (1, 0, 0, 1, 0),
    "negative_type": (0, -1, 0, 1, 0),
    "type_beyond_basis_rows": (0, 1, 0, 1, 0),
    "negative_first_split": (0, 0, -1, 1, 0),
    "decreasing_splits": (0, 0, 1, 0, 0),
    "split_beyond_edges": (0, 0, 0, 2, 0),
    "negative_edge_id": (0, 0, 0, 1, -1),
    "edge_id_beyond_edges": (0, 0, 0, 1, 1),
}


def _devices():
    return ["/CPU:0"] + (["/GPU:0"] if tf.config.get_visible_devices("GPU") else [])


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("index_dtype", [tf.uint32, tf.int64])
@pytest.mark.parametrize("case", sorted(INVALID_ONE_EDGE))
@pytest.mark.parametrize("gradient", [False, True])
def test_out_of_range_metadata_raises_before_indexing(device, index_dtype, case, gradient):
    from bmtk.simulator.dpointnet.custom_ops.master_projection import library
    native = library()
    spikes = tf.ones((2, 1), tf.float32)
    weights = tf.constant([0.75], tf.float32)
    basis = tf.constant([[1.0, 2.0, 3.0, 4.0]], tf.float32)
    dy = tf.ones((2, 4), tf.float32)

    def run(values):
        # Negative IDs wrap to huge uint32 values, which the same bounds reject.
        metadata = tf.cast(tf.constant(values, tf.int64), index_dtype)
        if gradient:
            return native.dpointnet_master_projection_grad(
                spikes, weights, metadata, basis, dy, n_post=1, n_edges=1,
            )
        return native.dpointnet_master_projection(
            spikes, weights, metadata, basis, n_post=1, n_edges=1,
        )

    with tf.device(device):
        with pytest.raises(tf.errors.InvalidArgumentError, match="Invalid master projection metadata"):
            run(INVALID_ONE_EDGE[case])
        # The device remains usable: valid metadata still matches the oracle.
        result = run(VALID_ONE_EDGE)
    if gradient:
        np.testing.assert_allclose(result[0], np.full((2, 1), 0.75 * 10.0))
        np.testing.assert_allclose(result[1], [2 * 10.0])
    else:
        np.testing.assert_allclose(result, np.tile([[0.75, 1.5, 2.25, 3.0]], (2, 1)))


@pytest.mark.parametrize("device", _devices())
def test_connectivity_type_beyond_the_basis_rows_is_rejected(device):
    conn = cpu_connectivity([[0, 0]], [1], 1, 1)
    with tf.device(device):
        with pytest.raises(tf.errors.InvalidArgumentError, match="synapse type|Invalid master"):
            project(
                tf.ones((1, 1), tf.float32), tf.ones((1,), tf.float32),
                conn["master_metadata"], tf.ones((1, 4), tf.float32), 1, 1,
            )
