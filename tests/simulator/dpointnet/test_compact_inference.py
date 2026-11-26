import copy
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.core.simulation_config import SimulationConfig
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
from bmtk.simulator.dpointnet.compact_inference import prepare_recurrent, resolve_connectivity
from bmtk.simulator.dpointnet.custom_ops import compact_inference_ops
from bmtk.simulator.dpointnet.network_adaptor import NetworkAdaptor
from bmtk.simulator.dpointnet.rnn_model import RNN
from test_precision_credit import make_cell


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def resolve(requested="auto", **options):
    kwargs = dict(inference_only=True, cell_cls=GLIF3Cell, params={},
                  strategy=SimpleNamespace(num_replicas_in_sync=1))
    kwargs.update(options)
    return resolve_connectivity(requested, **kwargs)


def variable_device(variable):
    value = getattr(variable, "value", variable)
    return (value() if callable(value) else value).device


@pytest.mark.parametrize("options,reason", [
    ({"inference_only": False}, "training"),
    ({"cell_cls": type("Custom", (GLIF3Cell,), {})}, "custom cell"),
    ({"strategy": SimpleNamespace(num_replicas_in_sync=2)}, "one replica"),
    ({"params": {"train_recurrent_per_type": True}}, "per-type"),
    ({"params": {"use_pair_projection": True}}, "explicit recurrent"),
    ({"params": {"use_fused_cuda": False}}, "override"),
    ({"params": {"temporal_gradient_precision": "float32"}}, "temporal"),
    ({"params": {"dynamics_mode": "legacy"}}, "legacy"),
    ({"compute_dtype": tf.float64}, "precision"),
    ({"neuron_partitioned": True}, "neuron-partitioned"),
])
def test_auto_retains_supported_standard_path_and_explicit_compact_errors(options, reason):
    report = resolve(**options)
    assert report["selected"] == "standard"
    assert reason in report["reason"]
    with pytest.raises(ValueError, match="Compact inference requested"):
        resolve("compact", **options)


def test_missing_operator_is_reported_without_install_or_silent_explicit_fallback(monkeypatch):
    monkeypatch.setattr(compact_inference_ops, "available", lambda: False)
    monkeypatch.setattr(compact_inference_ops, "status", lambda: "operator missing")
    assert resolve()["reason"] == "operator missing"
    with pytest.raises(ValueError, match="operator missing"):
        resolve("compact")
    assert resolve("standard")["selected"] == "standard"
    for value in (True, None, "fast"):
        with pytest.raises(ValueError, match="inference_connectivity"):
            resolve(value)


@pytest.mark.parametrize("value", ["auto", "invalid", None, 0, 1, True])
def test_explicit_recurrent_options_are_not_overwritten_by_backend_selection(value):
    options = {"use_active_row_forward": value}
    assert resolve(params=options)["selected"] == "standard"
    with pytest.raises(ValueError, match="explicit recurrent"):
        resolve("compact", params=options)


def test_enclosing_strategy_retains_standard_instead_of_placing_host_masters_on_gpu(monkeypatch):
    monkeypatch.setattr(tf.distribute, "has_strategy", lambda: True)
    assert "enclosing strategy" in resolve()["reason"]
    with pytest.raises(ValueError, match="enclosing strategy"):
        resolve("compact")


def test_gpu_selection_reads_the_compact_library_architecture_manifest(monkeypatch, tmp_path):
    device = SimpleNamespace(name="GPU:0")
    monkeypatch.setattr(compact_inference_ops, "_PATH", tmp_path / "compact.so")
    monkeypatch.setattr(compact_inference_ops, "_OPS", SimpleNamespace())
    monkeypatch.setattr(tf.config, "get_visible_devices", lambda kind: [device])
    monkeypatch.setattr(tf.config.experimental, "get_device_details",
                        lambda gpu: {"compute_capability": (8, 6)})
    monkeypatch.setattr(compact_inference_ops.csr_spike_ops, "_selected_physical_gpus", lambda: [device])
    monkeypatch.setattr(compact_inference_ops.csr_spike_ops, "_environment_flag", lambda name: False)
    metadata = tmp_path / "compact.archs"
    metadata.write_text("sm=86\nptx=86\n")
    assert compact_inference_ops.available()
    metadata.write_text("sm=90\nptx=90\n")
    assert not compact_inference_ops.available()
    assert "incompatible" in compact_inference_ops.status()


def compact_cell(selective=False, batch=2, noise=False, **overrides):
    network, inputs, params = make_cell(
        mode="nest", selective=selective, return_spec=True, noise=noise,
    )
    params.update(train_recurrent=False, batch_size=batch, **overrides)
    host = prepare_recurrent(network, params, params.get("dt", 1.0),
                             tf.float16 if selective else tf.float32, tf.float32)
    compact = GLIF3Cell(network, inputs, _compact_recurrent=host, **params)
    standard = GLIF3Cell(network, inputs, **params)
    return compact, standard


native = pytest.mark.skipif(not compact_inference_ops.available(), reason="Build compact inference operators")


@native
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("batch", [1, 3])
def test_projection_matches_independent_canonical_oracle_eager_and_compiled(selective, batch):
    compact, _ = compact_cell(selective, batch)
    dtype = tf.float16 if selective else tf.float32
    spikes = tf.constant(np.tile([1, 2, 3, 4], (batch, 1)), dtype)
    expected = np.array([[6., 1.8, .6, .3], [-1., -.3, -.1, -.05]] * batch)
    for project in (compact.calculate_i_rec_with_custom_grad,
                    tf.function(compact.calculate_i_rec_with_custom_grad)):
        actual = project(spikes).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3 if selective else 1e-6, atol=1e-3 if selective else 1e-7)
        assert actual.dtype == dtype.as_numpy_dtype
    assert compact.recurrent_indices is None
    assert compact.recurrent_fused_connectivity is None
    assert compact.recurrent_csr_weight_values_compute is None
    assert compact.compact_row_ptr.dtype == tf.int64
    assert compact.compact_targets.dtype == tf.uint32
    assert "CPU" in variable_device(compact.recurrent_weight_values)
    assert "CPU" in compact._compact_recurrent.order.device
    assert compact.recurrent_weight_values_compute.shape == (2,)


@native
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("hard_reset", [False, True])
def test_full_state_noise_and_chunk_carry_match_standard(selective, noise, hard_reset):
    compact, standard = compact_cell(selective, noise=noise, noise_seed=37, hard_reset=hard_reset)
    dtype = tf.float16 if selective else tf.float32
    inputs = tf.constant(np.tile([[1., 0.], [0., 2.], [1., 1.]], (2, 8, 1)), dtype)
    initial = compact.zero_state(2, dtype)
    runner = ExplicitStateRNN(compact, return_sequences=True, return_state=True)
    reference = ExplicitStateRNN(standard, return_sequences=True, return_state=True)
    expected = reference(inputs, initial_state=initial)
    compiled = tf.function(lambda x, s: runner(x, initial_state=s), input_signature=[
        tf.TensorSpec([2, None, 2], dtype),
        tuple(tf.TensorSpec(value.shape, value.dtype) for value in initial),
    ])
    actual = compiled(inputs, initial)
    first = compiled(inputs[:, :9], initial)
    second = compiled(inputs[:, 9:], first[1:])
    np.testing.assert_array_equal(expected[0][0], actual[0][0])
    for left, right in zip(tf.nest.flatten(expected), tf.nest.flatten(actual)):
        np.testing.assert_allclose(left, right, rtol=2e-3 if selective else 2e-6, atol=2e-3 if selective else 2e-6)
    for whole, chunk in zip(tf.nest.flatten(actual[1:]), tf.nest.flatten(second[1:])):
        np.testing.assert_array_equal(whole, chunk)
    for whole, early, late in zip(tf.nest.flatten(actual[0]), tf.nest.flatten(first[0]), tf.nest.flatten(second[0])):
        np.testing.assert_array_equal(whole, tf.concat([early, late], axis=1))
    with pytest.raises(ValueError, match="cannot execute training"):
        runner(inputs, initial_state=initial, training=True)


@native
def test_checkpoint_restore_refresh_and_canonical_values(tmp_path):
    compact, standard = compact_cell()
    expected = standard.recurrent_weight_values.numpy().copy()
    checkpoint = tf.train.Checkpoint(cell=compact)
    path = checkpoint.save(str(tmp_path / "weights"))
    compact.recurrent_weight_values.assign(np.zeros_like(expected))
    compact.refresh_recurrent_weight_shadow()
    np.testing.assert_array_equal(
        compact.calculate_i_rec_with_custom_grad(tf.ones([2, 4])), np.zeros([4, 4]))
    checkpoint.restore(path).assert_consumed()
    compact.refresh_recurrent_weight_shadow()
    np.testing.assert_array_equal(compact.recurrent_weight_values, expected)
    np.testing.assert_array_equal(
        compact.calculate_i_rec_with_custom_grad(tf.ones([2, 4])),
        standard.calculate_i_rec_with_custom_grad(tf.ones([2, 4])),
    )
    with tf.GradientTape() as tape:
        spikes = tf.Variable(tf.ones([2, 4]))
        value = tf.reduce_sum(compact.calculate_i_rec_with_custom_grad(spikes))
    with pytest.raises(ValueError, match="no adjoint"):
        tape.gradient(value, spikes)


class FixtureNetwork(NetworkAdaptor):
    def __init__(self, data, name, network_type):
        super().__init__(name, network_type)
        self.data = data

    @property
    def n_nodes(self):
        return self.data.get("n_nodes", self.data.get("n_inputs"))

    def to_dict(self):
        return self.data


def from_config(monkeypatch, connectivity="auto", training=None, dtype="float32", native_state=False):
    network, inputs, params = make_cell(
        mode="nest", selective=dtype == "float16", return_spec=True,
    )
    params.pop("acceleration_profile")
    params.pop("batch_size")
    if native_state:
        params.pop("use_fused_state")
    inputs["drive"]["input_type"] = "spikes"
    recurrent = FixtureNetwork(network, "recurrent", "recurrent")
    external = FixtureNetwork(inputs["drive"], "drive", "input")
    monkeypatch.setattr(NetworkAdaptor, "from_dict", lambda _: ([recurrent], [external]))
    config = SimulationConfig({
        "run": {"batch_size": 2, "seq_len": 3, "dtype": dtype,
                "inference_connectivity": connectivity},
        "rnn_cell_params": dict(params, cell_model="GLIF3"),
        "networks": {}, "components": {}, "inputs": {}, "training": training,
        "inference": {"inputs": []},
    })
    return RNN.from_config(config)


@native
def test_config_selects_compact_and_standard_override_preserves_values(monkeypatch):
    compact = from_config(monkeypatch)
    compact.build()
    assert compact.connectivity_report["selected"] == "compact"
    standard = from_config(monkeypatch, "standard")
    standard.build()
    assert standard.connectivity_report["selected"] == "standard"
    # Standard GPU auto may select CSR-master weights stored in live order; compare canonical order.
    np.testing.assert_array_equal(
        compact.cell.canonical_weight_values(), standard.cell.canonical_weight_values())
    assert "CPU" in variable_device(compact.cell.recurrent_weight_values)
    spikes = tf.constant([[[1., 0.], [0., 1.], [2., 1.]]] * 2)
    for left, right in zip(
        tf.nest.flatten(compact.run_extractor(spikes, compact.zero_state)),
        tf.nest.flatten(standard.run_extractor(spikes, standard.zero_state)),
    ):
        np.testing.assert_allclose(left, right, rtol=2e-6, atol=2e-6)
    with pytest.raises(ValueError, match="Cannot attach training"):
        compact.set_training()
    compact.cleanup()
    standard.cleanup()


@native
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_automatic_native_state_and_compact_projection_execute_together(monkeypatch, dtype):
    from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import fused_nest_state_available
    if not fused_nest_state_available():
        pytest.skip("This combination requires actual compatible native NEST state operators")
    compact = from_config(monkeypatch, dtype=dtype, native_state=True)
    compact.build()
    standard = from_config(monkeypatch, "standard", dtype=dtype, native_state=True)
    standard.build()
    assert compact.cell._use_fused_state and standard.cell._use_fused_state
    spikes = tf.constant([[[1., 0.], [0., 1.], [2., 1.]]] * 2, tf.as_dtype(dtype))
    compiled = tf.function(lambda x: compact.run_extractor(x, compact.zero_state))
    actual = compiled(spikes)
    expected = standard.run_extractor(spikes, standard.zero_state)
    np.testing.assert_array_equal(actual[0][0], expected[0][0])
    for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
        tolerance = 2e-3 if dtype == "float16" else 2e-6
        np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)
    graph = compiled.get_concrete_function(spikes).graph.as_graph_def()
    operations = {node.op for node in graph.node}
    operations.update(node.op for function in graph.library.function for node in function.node_def)
    assert "CompactInferenceCurrents" in operations
    assert any("Nest" in op and "State" in op for op in operations), sorted(operations)
    compact.cleanup()
    standard.cleanup()


@native
def test_training_config_and_inference_keep_trainable_standard_connectivity(monkeypatch):
    training = {
        "n_epochs": 1, "steps_per_epoch": 1, "learning_rate": 0.01,
        "optimizer": {"name": "sgd"},
        "parameters": [{"name": "condition", "inputs": [], "loss_functions": {}}],
    }
    rnn = from_config(monkeypatch, training=training)
    rnn.build()
    assert rnn.connectivity_report["selected"] == "standard"
    assert rnn.cell.recurrent_weight_values.trainable
    assert rnn.cell._compact_recurrent is None
    if not rnn.cell._csr_master_weights:
        # CSR-master weights are differentiated only through the joint current sweep.
        with tf.GradientTape() as tape:
            value = tf.reduce_sum(rnn.cell.calculate_i_rec_with_custom_grad(tf.ones([2, 4])))
        gradient = tape.gradient(value, rnn.cell.recurrent_weight_values)
        assert gradient is not None
        assert np.any(gradient.numpy() != 0)
    rnn.cleanup()
    explicit = from_config(monkeypatch, connectivity="compact", training=training)
    with pytest.raises(ValueError, match="training"):
        explicit.build()
    explicit._training_engine = None
    with pytest.raises(ValueError, match="training"):
        explicit.build()


@native
def test_invalid_explicit_flags_still_reach_existing_constructor_validation(monkeypatch):
    rnn = from_config(monkeypatch)
    rnn.cell_params["use_active_row_forward"] = "invalid"
    with pytest.raises((TypeError, ValueError), match="use_active_row_forward"):
        rnn.build()


@native
def test_duplicate_basis_rows_and_actual_compact_inventory():
    network, inputs, params = make_cell(mode="nest", selective=False, return_spec=True)
    count = 8192
    network["synapses"].update(
        indices=np.tile([[0, 1], [1, 0]], (count // 2, 1)),
        weights=np.tile([30., -20.], count // 2),
        delays=np.tile([2., 1.], count // 2),
        syn_ids=np.tile([0, 1], count // 2),
        dynamics_params={"basis_weights": [[1., .3, .1, .05], [1., .3, .1, .05]]},
    )
    params["train_recurrent"] = False
    host = prepare_recurrent(network, params, 1., tf.float32, tf.float32)
    assert host.n_types == 1
    cell = GLIF3Cell(network, inputs, _compact_recurrent=host, **params)
    inventory = (cell.compact_row_ptr, cell.compact_targets,
                 cell.recurrent_weight_values_compute, cell.compact_basis)
    actual_bytes = sum(int(np.prod(value.shape)) * value.dtype.size for value in inventory)
    assert actual_bytes == 8 * count + 8 * 5 + 4 * 4
    current = cell.calculate_i_rec_with_custom_grad(tf.ones([2, 4]))
    expected = np.tile([[1.5, .45, .15, .075], [-1., -.3, -.1, -.05]], (2, 1)) * (count // 2)
    np.testing.assert_allclose(current, expected, rtol=1e-4, atol=1e-4)


@native
def test_native_projection_uses_actual_gpu_when_visible():
    if not tf.config.get_visible_devices("GPU"):
        pytest.skip("GPU execution check requires a visible GPU")
    compact, _ = compact_cell()
    current = compact.calculate_i_rec_with_custom_grad(tf.ones([2, 4]))
    assert "GPU:0" in current.device
    for variable in (compact.compact_row_ptr, compact.compact_targets,
                     compact.recurrent_weight_values_compute):
        assert "GPU:0" in variable_device(variable)
    assert "CPU:0" in variable_device(compact.recurrent_weight_values)


@native
@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("bases", [1, 3, 7])
def test_generic_basis_width_and_batch_against_independent_values(dtype, bases):
    batch = 3
    basis = np.arange(1, 2 * bases + 1).reshape(2, bases) / 8
    actual = compact_inference_ops.currents(
        tf.constant([[1., 2., 0., 3.]] * batch, dtype),
        tf.constant([0, 2, 2, 3, 4], tf.int64),
        tf.constant([0, 3, 1, 2], tf.uint32),
        tf.constant([.5, -.25, 1., 2.], dtype), tf.constant(basis, dtype), 2,
    )
    expected = np.tile([.5 * basis[0], -.25 * basis[1] + 6 * basis[0]], (batch, 1))
    np.testing.assert_array_equal(actual, expected.astype(dtype.as_numpy_dtype))


@native
def test_empty_recurrent_graph_projects_zero_without_dummy_edges():
    network, inputs, params = make_cell(mode="nest", selective=False, return_spec=True)
    network["synapses"].update(
        indices=np.empty((0, 2), dtype=np.int64), weights=np.empty(0),
        delays=np.empty(0), syn_ids=np.empty(0, dtype=np.int64),
    )
    params["train_recurrent"] = False
    host = prepare_recurrent(network, params, 1., tf.float32, tf.float32)
    cell = GLIF3Cell(network, inputs, _compact_recurrent=host, **params)
    assert cell.recurrent_weight_values_compute.shape == (0,)
    np.testing.assert_array_equal(
        cell.calculate_i_rec_with_custom_grad(tf.ones([2, 2])), np.zeros([4, 4]))


@pytest.mark.parametrize("mutation", ["negative", "unknown_type", "delay", "nonfinite"])
def test_invalid_compact_inputs_raise_before_device_upload(mutation):
    network, _, params = make_cell(mode="nest", selective=False, return_spec=True)
    network = copy.deepcopy(network)
    if mutation == "negative":
        network["synapses"]["indices"] = np.array([[-1, 0], [1, 0]])
    elif mutation == "unknown_type":
        network["synapses"]["syn_ids"][0] = 99
    elif mutation == "delay":
        network["synapses"]["delays"][0] = np.nan
    else:
        network["synapses"]["weights"][0] = np.inf
    with pytest.raises(ValueError):
        prepare_recurrent(network, params, 1., tf.float32, tf.float32)


def test_compact_refresh_keeps_trainable_input_shadows_in_sync():
    compact, _ = compact_cell(selective=True)
    net = compact.inputs["drive"]
    master, shadow = net["input_weight_values"], net["input_weight_values_compute"]
    assert shadow is not master and master.trainable
    values = np.linspace(0.5, 1.5, int(master.shape[0])).astype(np.float32)
    master.assign(values)
    compact.refresh_recurrent_weight_shadow()
    np.testing.assert_array_equal(shadow.numpy(), values.astype(shadow.dtype.as_numpy_dtype))


@pytest.mark.parametrize("row_ptr, targets, message", [
    ([0, 2, 1], [0, 1], "row offsets"),
    ([1, 2, 2], [0, 1], "row offsets"),
    ([0, 1, 3], [0, 1], "row offsets"),
    ([0, 1, 2], [0, 4], "outside the neuron population"),
])
def test_connectivity_validation_rejects_invalid_offsets_and_targets(row_ptr, targets, message):
    from bmtk.simulator.dpointnet.compact_inference import validate_connectivity

    validate_connectivity([0, 1, 2], [0, 3], neurons=2, n_types=2, rows=2)
    with pytest.raises(ValueError, match=message):
        validate_connectivity(row_ptr, targets, neurons=2, n_types=2, rows=2)


@native
@pytest.mark.parametrize("row_ptr, targets", [([0, 1], [5]), ([0, 2], [0])])
def test_cpu_projection_reports_invalid_visited_connectivity(row_ptr, targets):
    with tf.device("/CPU:0"):
        with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
            compact_inference_ops.currents(
                tf.ones([1, 1]), tf.constant(row_ptr, tf.int64), tf.constant(targets, tf.uint32),
                tf.ones([len(targets)]), tf.ones([1, 4]), 1,
            ).numpy()
