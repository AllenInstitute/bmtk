import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.custom_ops import (
    build_csr_connectivity,
    fused_cuda_available,
    fused_spike_currents,
    reorder_csr_values,
)
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops
from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import (
    _csr_index_dtype,
    _resolve_packed_sm120_backward,
    _resolve_packed_sm120_model_option,
    _validate_packed_sm120_option,
)
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import (
    fused_dense_state,
    fused_glif_state_available,
    fused_spike_shift,
)
from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
    GLIF3Cell,
    _fused_cuda_dtype_error,
    _resolve_fused_state,
    _resolve_pair_projection,
    _validate_fixed4_forward_option,
    _validate_fused_cuda_option,
    _validate_pair_projection_option,
    spike_function,
)

INDICES = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 2],
        [0, 1],
    ],
    dtype=np.int64,
)
SYNAPSE_TYPES = np.array([0, 1, 0, 1], dtype=np.int64)


def test_tracked_master_weight_stays_fp32_inside_mixed_precision_call():
    old_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    class Probe(tf.keras.layers.Layer):
        _tracked_weight = GLIF3Cell._tracked_weight

        def build(self, _input_shape):
            self.master = self._tracked_weight(
                np.ones(3, np.float32), "master", True, tf.float32
            )

        def call(self, _inputs):
            return tf.convert_to_tensor(self.master)

    try:
        probe = Probe()
        with tf.GradientTape() as tape:
            value = probe(tf.ones([1], tf.float16))
            loss = tf.reduce_sum(value)
        gradient = tape.gradient(loss, probe.master)

        assert probe.master.dtype == "float32"
        assert value.dtype == tf.float32
        assert gradient.dtype == tf.float32
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)


@pytest.mark.parametrize("value", [True, False])
def test_fixed4_forward_option_accepts_booleans(value):
    assert _validate_fixed4_forward_option(value) is value


@pytest.mark.parametrize("value", [0, 1, "auto", np.bool_(True)])
def test_fixed4_forward_option_rejects_lookalikes(value):
    with pytest.raises(ValueError, match="use_fixed4_input_forward"):
        _validate_fixed4_forward_option(value)


def _reference_currents_for_connectivity(
    spikes, master_weights, basis, indices, synapse_types, n_post
):
    post_ids = tf.constant(indices[:, 0], tf.int32)
    pre_ids = tf.constant(indices[:, 1], tf.int32)
    edge_basis = tf.gather(basis, synapse_types)
    compute_weights = tf.cast(master_weights, spikes.dtype)
    batch_outputs = []
    for batch in range(spikes.shape[0]):
        edge_values = (
            tf.gather(spikes[batch], pre_ids)[:, tf.newaxis]
            * compute_weights[:, tf.newaxis]
            * edge_basis
        )
        batch_outputs.append(
            tf.math.unsorted_segment_sum(edge_values, post_ids, n_post)
        )
    return tf.reshape(tf.stack(batch_outputs), [-1, basis.shape[1]])


def _reference_currents(spikes, master_weights, basis):
    return _reference_currents_for_connectivity(
        spikes,
        master_weights,
        basis,
        INDICES,
        SYNAPSE_TYPES,
        n_post=2,
    )


def _reference_dense_state(
    prev_z,
    voltage,
    refractory,
    asc,
    psc_rise,
    psc,
    rec_inputs,
    parameters,
    hard_reset,
):
    batch_size = tf.shape(voltage)[0]
    neurons = voltage.shape[1]
    new_psc_rise = (
        psc_rise * parameters["syn_decay"] + rec_inputs * parameters["psc_initial"]
    )
    new_psc = (
        psc * parameters["syn_decay"]
        + parameters["dt"] * parameters["syn_decay"] * psc_rise
    )
    asc_3d = tf.reshape(asc, (batch_size, neurons, 2))
    new_asc = (
        parameters["asc_decay"] * asc_3d
        + tf.stop_gradient(prev_z)[..., None] * parameters["asc_amps"]
    )
    new_asc = tf.reshape(new_asc, tf.shape(asc))
    current = tf.reduce_sum(
        tf.reshape(psc, (batch_size, neurons, 4)), axis=-1
    ) + tf.reduce_sum(asc_3d, axis=-1)
    dampening = parameters["voltage_gradient_dampening"]
    dampened_voltage = voltage * (1.0 - dampening) + tf.stop_gradient(
        voltage * dampening
    )
    new_voltage = (
        parameters["decay"] * dampened_voltage
        + parameters["current_factor"] * current
        - tf.stop_gradient(prev_z)
    )
    new_refractory = tf.stop_gradient(
        tf.maximum(
            refractory
            + tf.cast(prev_z, refractory.dtype) * parameters["t_ref_steps"]
            - tf.cast(1, refractory.dtype),
            tf.cast(0, refractory.dtype),
        )
    )
    if hard_reset:
        new_voltage = tf.where(new_refractory > 0, parameters["v_reset"], new_voltage)
    return new_voltage, new_refractory, new_asc, new_psc_rise, new_psc


def _metadata_values(connectivity):
    return tf.raw_ops.ReadVariableOp(
        resource=connectivity["metadata_handle"],
        dtype=tf.dtypes.as_dtype(connectivity["index_dtype"]),
    )


def test_build_csr_connectivity_groups_edges_by_source():
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 3, 2, 2, build_compact_pairs=True
    )
    metadata = _metadata_values(connectivity).numpy()

    np.testing.assert_array_equal(metadata[:4], [0, 1, 0, 1])
    np.testing.assert_array_equal(metadata[4:8], [0, 1, 1, 0])
    np.testing.assert_array_equal(metadata[8:12], [0, 2, 3, 4])
    np.testing.assert_array_equal(metadata[12:16], [0, 1, 3, 2])
    np.testing.assert_array_equal(metadata[16:20], [0, 3, 1, 2])
    np.testing.assert_array_equal(metadata[20:24], [0, 0, 1, 1])
    np.testing.assert_array_equal(metadata[24:], [0, 1, 0, 1])
    assert connectivity["n_pairs"] == 4
    assert connectivity["index_dtype"] == "uint32"
    assert _metadata_values(connectivity).dtype == tf.uint32


def test_csr_index_dtype_retains_int64_beyond_uint32():
    uint32_max = np.iinfo(np.uint32).max

    assert _csr_index_dtype(uint32_max, 1, 1, 1) == tf.uint32
    assert _csr_index_dtype(uint32_max + 1, 1, 1, 1) == tf.int64


@pytest.mark.parametrize(
    ("indices", "synapse_types", "message"),
    [
        (np.array([[2, 0]]), np.array([0]), "Postsynaptic indices"),
        (np.array([[0, 0]]), np.array([2]), "Synapse type indices"),
    ],
)
def test_build_csr_connectivity_rejects_unsafe_indices(indices, synapse_types, message):
    with pytest.raises(ValueError, match=message):
        build_csr_connectivity(indices, synapse_types, 1, 2, 2)


@pytest.mark.parametrize(
    ("indices", "synapse_types"),
    [
        (np.array([[0.0, 0.5]]), np.array([0])),
        (np.array([[0, 0]]), np.array([0.5])),
        (np.array([[0, 0]]), np.array([np.nan])),
    ],
)
def test_build_csr_connectivity_rejects_non_integer_metadata(indices, synapse_types):
    with pytest.raises(ValueError, match="finite integer values"):
        build_csr_connectivity(indices, synapse_types, 1, 1, 1)


@pytest.mark.parametrize(
    ("indices", "synapse_types", "n_source_neurons", "n_synapse_types"),
    [
        (
            np.array([[0, np.iinfo(np.uint64).max]], dtype=np.uint64),
            np.array([0]),
            int(np.iinfo(np.uint64).max) + 1,
            1,
        ),
        (
            np.array([[0, 0]]),
            np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
            1,
            int(np.iinfo(np.uint64).max) + 1,
        ),
    ],
)
def test_build_csr_connectivity_rejects_metadata_outside_int64(
    indices, synapse_types, n_source_neurons, n_synapse_types
):
    with pytest.raises(ValueError, match="int64 range"):
        build_csr_connectivity(
            indices,
            synapse_types,
            n_source_neurons,
            1,
            n_synapse_types,
        )


@pytest.mark.parametrize(
    "value",
    [
        "auto",
        np.str_("auto"),
        b"auto",
        np.bytes_("auto"),
        np.array("auto"),
        np.array(b"auto"),
    ],
)
def test_fused_cuda_option_accepts_string_scalars(value):
    assert _validate_fused_cuda_option(value) == "auto"


@pytest.mark.parametrize("value", [0, 1, np.bool_(False), np.bool_(True)])
def test_fused_cuda_option_rejects_non_boolean_lookalikes(value):
    with pytest.raises(ValueError, match="use_fused_cuda"):
        _validate_fused_cuda_option(value)


@pytest.mark.parametrize(
    "value",
    [
        "auto",
        np.str_("auto"),
        b"auto",
        np.bytes_("auto"),
        np.array("auto"),
        np.array(b"auto"),
        True,
        False,
    ],
)
def test_pair_projection_option_accepts_auto_and_booleans(value):
    expected = value.item() if isinstance(value, np.ndarray) else value
    if isinstance(expected, bytes):
        expected = expected.decode("utf-8")
    assert _validate_pair_projection_option(value) == expected


@pytest.mark.parametrize("value", [0, 1, np.bool_(False), np.bool_(True), "yes"])
def test_pair_projection_option_rejects_lookalikes(value):
    with pytest.raises(ValueError, match="use_pair_projection"):
        _validate_pair_projection_option(value)


@pytest.mark.parametrize(
    "value",
    [
        "auto",
        np.str_("auto"),
        b"auto",
        np.bytes_("auto"),
        np.array("auto"),
        np.array(b"auto"),
        True,
        False,
    ],
)
def test_packed_sm120_option_accepts_auto_and_booleans(value):
    expected = value.item() if isinstance(value, np.ndarray) else value
    if isinstance(expected, bytes):
        expected = expected.decode("utf-8")
    assert _validate_packed_sm120_option(value) == expected


@pytest.mark.parametrize("value", [0, 1, np.bool_(False), np.bool_(True), "yes"])
def test_packed_sm120_option_rejects_lookalikes(value):
    with pytest.raises(ValueError, match="use_packed_sm120_backward"):
        _validate_packed_sm120_option(value)


def test_packed_sm120_auto_uses_fallback_on_sm80(monkeypatch):
    connectivity = {
        "index_dtype": "uint32",
        "n_pairs": 1,
    }
    monkeypatch.setattr(csr_spike_ops, "_gpu_compute_architecture", lambda: 80)

    assert not _resolve_packed_sm120_backward(
        "auto", tf.ones([32, 3], tf.float16), connectivity, tf.ones([2, 4])
    )


def test_packed_sm120_auto_selects_eligible_sm120(monkeypatch):
    connectivity = {
        "index_dtype": "uint32",
        "n_pairs": 1,
    }
    monkeypatch.setattr(csr_spike_ops, "_gpu_compute_architecture", lambda: 120)

    assert _resolve_packed_sm120_backward(
        "auto", tf.ones([32, 3], tf.float16), connectivity, tf.ones([2, 4])
    )


@pytest.mark.parametrize(
    "connectivity",
    [
        {"index_dtype": "uint32", "n_pairs": 0},
        {"index_dtype": "int64", "n_pairs": 1},
    ],
)
def test_packed_sm120_external_auto_falls_back_for_connectivity(
    monkeypatch, connectivity
):
    monkeypatch.setattr(csr_spike_ops, "_gpu_compute_architecture", lambda: 120)

    assert not _resolve_packed_sm120_backward(
        "auto", tf.ones([32, 3], tf.float16), connectivity, tf.ones([2, 4])
    )


def test_forced_packed_sm120_rejects_sm80(monkeypatch):
    connectivity = {
        "index_dtype": "uint32",
        "n_pairs": 1,
    }
    monkeypatch.setattr(csr_spike_ops, "_gpu_compute_architecture", lambda: 80)

    with pytest.raises(ValueError, match="SM80"):
        _resolve_packed_sm120_backward(
            True, tf.ones([32, 3], tf.float16), connectivity, tf.ones([2, 4])
        )


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [(80, False), (86, "auto"), (89, "auto"), (120, "auto")],
)
def test_packed_sm120_external_model_selection(monkeypatch, architecture, expected):
    monkeypatch.setattr(
        csr_spike_ops, "_gpu_compute_architecture", lambda: architecture
    )

    assert (
        _resolve_packed_sm120_model_option("auto", True, tf.float16, 32, 4) is expected
    )


def test_forced_packed_sm120_external_model_rejects_sm80(monkeypatch):
    monkeypatch.setattr(csr_spike_ops, "_gpu_compute_architecture", lambda: 80)

    with pytest.raises(ValueError, match="use_packed_sm120_external_backward.*SM80"):
        _resolve_packed_sm120_model_option(True, True, tf.float16, 32, 4)


def test_fixed_input_connectivity_omits_compact_pair_metadata():
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 3, 2, 2, build_compact_pairs=False
    )

    assert connectivity["n_pairs"] == 0
    assert _metadata_values(connectivity).shape[0] == 3 * len(INDICES) + 3 + 1


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_currents_forced_packed_sm120_rejects_sm80(monkeypatch):
    monkeypatch.setattr(csr_spike_ops, "_gpu_compute_architecture", lambda: 80)
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 3, 2, 2, build_compact_pairs=True
    )
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, tf.float16), connectivity)

    with pytest.raises(ValueError, match="SM80"):
        fused_spike_currents(
            tf.ones([32, 3], tf.float16),
            master_weights,
            csr_weights,
            connectivity,
            tf.ones([2, 4], tf.float16),
            n_post=2,
            compute_spike_gradient=True,
            use_packed_sm120_backward=True,
        )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_external_forced_packed_sm120_rejects_sm80(monkeypatch):
    monkeypatch.setattr(csr_spike_ops, "_gpu_compute_architecture", lambda: 80)
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 3, 2, 2, build_compact_pairs=True
    )
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, tf.float16), connectivity)

    with pytest.raises(ValueError, match="SM80"):
        fused_spike_currents(
            tf.ones([32, 3], tf.float16),
            master_weights,
            csr_weights,
            connectivity,
            tf.ones([2, 4], tf.float16),
            n_post=2,
            compute_spike_gradient=False,
            use_packed_sm120_backward=True,
        )


@pytest.mark.parametrize(
    ("option", "fused_cuda", "batch_size", "n_syn_basis", "expected"),
    [
        ("auto", True, 32, 4, True),
        ("auto", True, 5, 4, False),
        ("auto", True, 32, 5, False),
        ("auto", False, 32, 4, False),
        (False, True, 32, 4, False),
        (False, False, 5, 5, False),
        (True, True, 32, 4, True),
    ],
)
def test_pair_projection_policy_resolution(
    option, fused_cuda, batch_size, n_syn_basis, expected
):
    assert (
        _resolve_pair_projection(option, fused_cuda, batch_size, n_syn_basis)
        is expected
    )


@pytest.mark.parametrize(
    ("option", "available", "n_syn_basis", "pseudo_gauss", "expected"),
    [
        (False, True, 4, False, False),
        ("auto", True, 4, False, True),
        ("auto", False, 4, False, False),
        ("auto", True, 3, False, False),
        ("auto", True, 4, True, False),
    ],
)
def test_fused_state_policy_resolution(
    monkeypatch, option, available, n_syn_basis, pseudo_gauss, expected
):
    monkeypatch.setattr(
        "bmtk.simulator.dpointnet.cell_models.glif3_cell.fused_glif_state_available",
        lambda: available,
    )
    assert _resolve_fused_state(option, n_syn_basis, pseudo_gauss) is expected


def test_forced_fused_state_rejects_incompatible_model(monkeypatch):
    monkeypatch.setattr(
        "bmtk.simulator.dpointnet.cell_models.glif3_cell.fused_glif_state_available",
        lambda: True,
    )
    with pytest.raises(ValueError, match="pseudo_gauss"):
        _resolve_fused_state(True, 4, True)


@pytest.mark.parametrize(
    ("fused_cuda", "batch_size", "n_syn_basis", "message"),
    [
        (False, 32, 4, "fused CUDA"),
        (True, 5, 4, "batch_size is 5"),
        (True, 32, 5, "basis has 5 columns"),
    ],
)
def test_forced_pair_projection_rejects_incompatible_models(
    fused_cuda, batch_size, n_syn_basis, message
):
    with pytest.raises(ValueError, match=message):
        _resolve_pair_projection(True, fused_cuda, batch_size, n_syn_basis)


def test_fused_cuda_dtype_error_describes_unsupported_policy():
    assert _fused_cuda_dtype_error(tf.float16, tf.float32) is None
    message = _fused_cuda_dtype_error(tf.bfloat16, tf.float32)
    assert "compute_dtype=bfloat16" in message
    assert "variable_dtype=float32" in message


def test_fused_availability_respects_visible_devices(monkeypatch):
    monkeypatch.setattr(tf.config, "get_visible_devices", lambda device_type: [])

    assert not csr_spike_ops.fused_cuda_available()


def test_fused_availability_rejects_multiple_visible_gpus(monkeypatch):
    monkeypatch.setattr(
        tf.config, "get_visible_devices", lambda device_type: [object(), object()]
    )

    assert not csr_spike_ops.fused_cuda_available()


def test_fused_availability_rejects_incompatible_gpu(monkeypatch):
    gpu = object()
    monkeypatch.setattr(tf.config, "get_visible_devices", lambda device_type: [gpu])
    monkeypatch.setattr(
        tf.config.experimental,
        "get_device_details",
        lambda device: {"compute_capability": (6, 0)},
    )

    assert not csr_spike_ops.fused_cuda_available()


@pytest.mark.skipif(
    not fused_glif_state_available(), reason="Fused GLIF state op is unavailable."
)
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("hard_reset", [False, True])
def test_fused_glif_state_matches_forward_and_gradients(dtype, hard_reset):
    rng = np.random.default_rng(43)
    batch_size = 2
    neurons = 3
    basis_width = neurons * 4

    def variable(shape, low=-0.2, high=0.2):
        return tf.Variable(rng.uniform(low, high, shape).astype(dtype.as_numpy_dtype))

    prev_z = tf.Variable(np.array([[0, 1, 0], [1, 0, 1]], dtype=dtype.as_numpy_dtype))
    voltage = variable((batch_size, neurons))
    asc = variable((batch_size, neurons * 2))
    psc_rise = variable((batch_size, basis_width))
    psc = variable((batch_size, basis_width))
    rec_inputs = variable((batch_size, basis_width))
    refractory = tf.constant([[0, 2, 1], [2, 0, 1]], tf.int16)
    parameters = {
        "syn_decay": tf.constant(rng.uniform(0.7, 0.95, (1, basis_width)), dtype),
        "psc_initial": tf.constant(rng.uniform(0.1, 0.5, (1, basis_width)), dtype),
        "asc_decay": tf.constant(rng.uniform(0.6, 0.9, (1, neurons, 2)), dtype),
        "asc_amps": tf.constant(rng.uniform(-0.2, 0.2, (1, neurons, 2)), dtype),
        "decay": tf.constant(rng.uniform(0.8, 0.95, (1, neurons)), dtype),
        "current_factor": tf.constant(rng.uniform(0.05, 0.2, (1, neurons)), dtype),
        "t_ref_steps": tf.constant([2, 3, 2], tf.int16),
        "dt": tf.constant(1.0, dtype),
        "v_reset": tf.constant(-0.1, dtype),
        "voltage_gradient_dampening": tf.constant(0.35, dtype),
    }
    variables = (prev_z, voltage, asc, psc_rise, psc, rec_inputs)
    upstream = tuple(
        tf.constant(rng.uniform(-0.5, 0.5, value.shape), dtype)
        for value in (voltage, refractory, asc, psc_rise, psc)
    )

    def evaluate(fused):
        with tf.GradientTape() as tape:
            if fused:
                outputs = fused_dense_state(
                    prev_z,
                    voltage,
                    refractory,
                    asc,
                    psc_rise,
                    psc,
                    rec_inputs,
                    syn_decay=parameters["syn_decay"],
                    psc_initial=parameters["psc_initial"],
                    asc_decay=parameters["asc_decay"],
                    asc_amps=parameters["asc_amps"],
                    decay=parameters["decay"],
                    current_factor=parameters["current_factor"],
                    t_ref_steps=parameters["t_ref_steps"],
                    dt=parameters["dt"],
                    v_reset=parameters["v_reset"],
                    voltage_gradient_dampening=parameters["voltage_gradient_dampening"],
                    hard_reset=hard_reset,
                )
            else:
                outputs = _reference_dense_state(
                    prev_z,
                    voltage,
                    refractory,
                    asc,
                    psc_rise,
                    psc,
                    rec_inputs,
                    parameters,
                    hard_reset,
                )
            loss = tf.add_n(
                [
                    tf.reduce_sum(output * weight)
                    for output, weight in zip(outputs, upstream)
                    if output.dtype.is_floating
                ]
            )
        return outputs, tape.gradient(loss, variables)

    fused_outputs, fused_gradients = evaluate(True)
    reference_outputs, reference_gradients = evaluate(False)
    tolerance = 3e-3 if dtype == tf.float16 else 1e-6
    for fused, reference in zip(fused_outputs, reference_outputs):
        np.testing.assert_allclose(fused, reference, rtol=tolerance, atol=tolerance)
    for variable, fused, reference in zip(
        variables, fused_gradients, reference_gradients
    ):
        if reference is None:
            reference = tf.zeros_like(variable)
        np.testing.assert_allclose(fused, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(
    not fused_glif_state_available(), reason="Fused GLIF state op is unavailable."
)
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_fused_spike_shift_matches_forward_and_gradients(dtype):
    voltage = tf.Variable([[-1.2, -0.1, 0.2], [0.0, 0.6, 1.4]], dtype=dtype)
    history = tf.Variable([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=dtype)
    refractory = tf.constant([[False, True, False], [False, False, True]])
    dampening = tf.constant(0.3, dtype)
    spike_upstream = tf.constant([[0.5, -0.2, 0.7], [0.1, 0.4, -0.3]], dtype)
    history_upstream = tf.constant(
        [[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]],
        dtype,
    )

    def evaluate(fused):
        with tf.GradientTape() as tape:
            if fused:
                spikes, new_history = fused_spike_shift(
                    voltage, refractory, history, dampening
                )
            else:
                spikes = spike_function(voltage, dampening)
                spikes = tf.where(refractory, tf.zeros_like(spikes), spikes)
                new_history = tf.concat([spikes, history[:, :-3]], axis=1)
            loss = tf.reduce_sum(spikes * spike_upstream) + tf.reduce_sum(
                new_history * history_upstream
            )
        return (spikes, new_history), tape.gradient(loss, (voltage, history))

    fused_outputs, fused_gradients = evaluate(True)
    reference_outputs, reference_gradients = evaluate(False)
    tolerance = 2e-3 if dtype == tf.float16 else 1e-6
    for fused, reference in zip(fused_outputs, reference_outputs):
        np.testing.assert_allclose(fused, reference, rtol=tolerance, atol=tolerance)
    for fused, reference in zip(fused_gradients, reference_gradients):
        np.testing.assert_allclose(fused, reference, rtol=tolerance, atol=tolerance)


def test_csr_connectivity_close_releases_resource():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    handle = connectivity["metadata_handle"]

    connectivity.close()

    with pytest.raises((tf.errors.NotFoundError, tf.errors.FailedPreconditionError)):
        tf.raw_ops.ReadVariableOp(resource=handle, dtype=tf.uint32)


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize(
    "spike_values",
    [
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
)
def test_fused_recurrent_currents_match_forward_and_gradients(dtype, spike_values):
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, dtype), connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=dtype)
    spikes = tf.Variable(spike_values, dtype=dtype)

    with tf.GradientTape() as fused_tape:
        fused = fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )
        fused_loss = tf.reduce_sum(fused)
    fused_gradients = fused_tape.gradient(fused_loss, [spikes, master_weights])

    with tf.GradientTape() as reference_tape:
        reference = _reference_currents(spikes, master_weights, basis)
        reference_loss = tf.reduce_sum(reference)
    reference_gradients = reference_tape.gradient(
        reference_loss, [spikes, master_weights]
    )

    tolerance = 2e-3 if dtype == tf.float16 else 1e-6
    np.testing.assert_allclose(
        fused.numpy(), reference.numpy(), rtol=tolerance, atol=tolerance
    )
    for fused_gradient, reference_gradient in zip(fused_gradients, reference_gradients):
        np.testing.assert_allclose(
            fused_gradient.numpy(),
            reference_gradient.numpy(),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_recurrent_gradient_scale_only_affects_spike_gradient():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    def currents_and_gradients(scale):
        spikes = tf.Variable([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            currents = fused_spike_currents(
                spikes,
                master_weights,
                csr_weights,
                connectivity,
                basis,
                n_post=2,
                compute_spike_gradient=True,
                spike_gradient_scale=scale,
            )
            loss = tf.reduce_sum(currents)
        gradients = tape.gradient(loss, (spikes, master_weights))
        return currents.numpy(), tuple(gradient.numpy() for gradient in gradients)

    full = currents_and_gradients(1.0)
    dampened = currents_and_gradients(0.25)

    np.testing.assert_allclose(dampened[0], full[0])
    np.testing.assert_allclose(dampened[1][0], full[1][0] * 0.25)
    np.testing.assert_allclose(dampened[1][1], full[1][1])


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("packed_option", [False, "auto"])
def test_pair_projected_batch32_matches_forward_and_gradients(dtype, packed_option):
    batch_size = 32
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 3, 2, 2, build_compact_pairs=True
    )
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, dtype), connectivity)
    basis = tf.constant(
        [[1.0, 0.5, 0.25, 0.125], [0.25, 2.0, 0.75, 1.5]],
        dtype=dtype,
    )
    spike_values = np.zeros((batch_size, 3), dtype=np.float32)
    spike_values[::2, 0] = 1.0
    spike_values[1::3, 1] = 2.0
    spike_values[2::5, 2] = 3.0
    spikes = tf.Variable(spike_values, dtype=dtype)
    loss_weights = tf.reshape(
        tf.cast(tf.range(1, batch_size * 2 * 4 + 1), dtype),
        [batch_size * 2, 4],
    ) / tf.cast(batch_size * 2 * 4, dtype)

    with tf.GradientTape() as fused_tape:
        fused = fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
            use_packed_sm120_backward=packed_option,
        )
        fused_loss = tf.reduce_sum(fused * loss_weights)
    fused_gradients = fused_tape.gradient(fused_loss, [spikes, master_weights])

    with tf.GradientTape() as reference_tape:
        reference = _reference_currents(spikes, master_weights, basis)
        reference_loss = tf.reduce_sum(reference * loss_weights)
    reference_gradients = reference_tape.gradient(
        reference_loss, [spikes, master_weights]
    )

    tolerance = 2e-2 if dtype == tf.float16 else 1e-4
    np.testing.assert_allclose(
        fused.numpy(), reference.numpy(), rtol=tolerance, atol=tolerance
    )
    for fused_gradient, reference_gradient in zip(fused_gradients, reference_gradients):
        np.testing.assert_allclose(
            fused_gradient.numpy(),
            reference_gradient.numpy(),
            rtol=tolerance,
            atol=tolerance,
        )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_pair_projected_batch32_writes_empty_rows_and_all_canonical_edges():
    batch_size = 32
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 5, 2, 2, build_compact_pairs=True
    )
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, tf.float16), connectivity)
    basis = tf.constant(
        [[1.0, 0.5, 0.25, 0.125], [0.25, 2.0, 0.75, 1.5]],
        dtype=tf.float16,
    )
    spikes = tf.Variable(
        np.random.default_rng(47).uniform(size=(batch_size, 5)).astype(np.float16)
    )
    loss_weights = tf.reshape(
        tf.cast(tf.range(1, batch_size * 2 * 4 + 1), tf.float16),
        [batch_size * 2, 4],
    ) / tf.cast(batch_size * 2 * 4, tf.float16)

    with tf.GradientTape() as tape:
        currents = fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
            use_packed_sm120_backward=False,
        )
        loss = tf.reduce_sum(currents * loss_weights)
    spike_gradient, weight_gradient = tape.gradient(loss, [spikes, master_weights])

    with tf.GradientTape() as tape:
        reference = _reference_currents(spikes, master_weights, basis)
        reference_loss = tf.reduce_sum(reference * loss_weights)
    reference_spike_gradient, reference_weight_gradient = tape.gradient(
        reference_loss, [spikes, master_weights]
    )

    np.testing.assert_array_equal(spike_gradient[:, 3:], 0.0)
    np.testing.assert_allclose(
        spike_gradient, reference_spike_gradient, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        weight_gradient, reference_weight_gradient, rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_input_currents_preserve_counts_and_only_differentiate_weights():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)
    spike_counts = tf.Variable([[2.0, 0.0, 3.0], [0.0, 4.0, 0.0]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        currents = fused_spike_currents(
            spike_counts,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=False,
        )
        loss = tf.reduce_sum(currents)
    spike_gradient, weight_gradient = tape.gradient(
        loss, [spike_counts, master_weights]
    )

    with tf.GradientTape() as reference_tape:
        reference = _reference_currents(spike_counts, master_weights, basis)
        reference_loss = tf.reduce_sum(reference)
    reference_weight_gradient = reference_tape.gradient(reference_loss, master_weights)

    assert spike_gradient is None
    np.testing.assert_allclose(currents.numpy(), reference.numpy())
    np.testing.assert_allclose(
        weight_gradient.numpy(), reference_weight_gradient.numpy()
    )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("use_fixed4", [False, True])
def test_fixed4_input_forward_matches_values_and_canonical_weight_gradient(
    dtype, use_fixed4
):
    indices = np.array(
        [
            [1, 2],
            [0, 1],
            [1, 0],
            [0, 2],
            [1, 1],
            [0, 0],
            [1, 2],
            [0, 1],
        ],
        dtype=np.int64,
    )
    synapse_types = np.array([1, 0, 0, 1, 1, 0, 0, 1], dtype=np.int64)
    connectivity = build_csr_connectivity(
        indices,
        synapse_types,
        3,
        2,
        2,
        build_compact_pairs=True,
        build_fixed4_incoming=True,
    )
    master_weights = tf.Variable([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, dtype), connectivity)
    basis = tf.constant([[1.0, 0.5, 0.25, 0.125], [0.25, 2.0, 0.75, 1.5]], dtype)
    activity = tf.constant(np.random.default_rng(67).poisson(0.5, size=(32, 3)), dtype)
    loss_weights = tf.reshape(tf.range(1, 32 * 2 * 4 + 1, dtype=tf.float32), [64, 4])
    loss_weights = tf.cast(loss_weights / tf.reduce_max(loss_weights), dtype)

    with tf.GradientTape() as tape:
        actual = fused_spike_currents(
            activity,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=False,
            compute_weight_gradient=True,
            use_fixed4_forward=use_fixed4,
        )
        actual_loss = tf.reduce_sum(actual * loss_weights)
    actual_gradient = tape.gradient(actual_loss, master_weights)

    with tf.GradientTape() as tape:
        reference = _reference_currents_for_connectivity(
            activity,
            master_weights,
            basis,
            indices,
            synapse_types,
            n_post=2,
        )
        reference_loss = tf.reduce_sum(reference * loss_weights)
    reference_gradient = tape.gradient(reference_loss, master_weights)

    tolerance = 2e-2 if dtype == tf.float16 else 1e-6
    np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual_gradient, reference_gradient, rtol=tolerance, atol=tolerance
    )
    assert connectivity["fixed4_incoming"]


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fixed4_forward_supports_dynamic_batch_signature():
    indices = np.array(
        [[post, pre] for post in range(2) for pre in (0, 1, 2, 0)],
        dtype=np.int64,
    )
    synapse_types = np.arange(8, dtype=np.int64) % 2
    connectivity = build_csr_connectivity(
        indices,
        synapse_types,
        3,
        2,
        2,
        build_fixed4_incoming=True,
    )
    master_weights = tf.Variable(tf.range(1, 9, dtype=tf.float32))
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5, 0.25, 0.125], [0.25, 2.0, 0.75, 1.5]])

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float32)])
    def run(activity):
        return fused_spike_currents(
            activity,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=False,
            compute_weight_gradient=True,
            use_fixed4_forward=True,
        )

    activity = tf.ones([32, 3], tf.float32)
    np.testing.assert_allclose(
        run(activity),
        _reference_currents_for_connectivity(
            activity, master_weights, basis, indices, synapse_types, 2
        ),
    )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize(
    ("key", "values"),
    [
        ("incoming_pre_ids", [99] * 8),
        ("incoming_edge_ids", [99] * 8),
        ("incoming_types", [99] * 8),
    ],
)
def test_fixed4_forward_ignores_malformed_incoming_indices(key, values):
    indices = np.array(
        [[post, pre] for post in range(2) for pre in (0, 1, 2, 0)],
        dtype=np.int64,
    )
    synapse_types = np.arange(8, dtype=np.int64) % 2
    base_connectivity = build_csr_connectivity(
        indices,
        synapse_types,
        3,
        2,
        2,
        build_fixed4_incoming=True,
    )
    master_weights = tf.Variable(tf.ones([8], tf.float32))
    csr_weights = reorder_csr_values(master_weights, base_connectivity)
    connectivity = csr_spike_ops.CsrConnectivity(
        {**base_connectivity, key: tf.constant(values, tf.uint32)}
    )
    currents = fused_spike_currents(
        tf.ones([32, 3], tf.float32),
        master_weights,
        csr_weights,
        connectivity,
        tf.ones([2, 4], tf.float32),
        n_post=2,
        compute_spike_gradient=False,
        use_fixed4_forward=True,
    )

    np.testing.assert_array_equal(currents, 0.0)


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fixed4_forward_rejects_non_fixed4_connectivity():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)

    with pytest.raises(ValueError, match="exactly four incoming edges"):
        fused_spike_currents(
            tf.ones([32, 3], tf.float32),
            tf.ones([4], tf.float32),
            tf.ones([4], tf.float32),
            connectivity,
            tf.ones([2, 4], tf.float32),
            n_post=2,
            compute_spike_gradient=False,
            use_fixed4_forward=True,
        )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fixed_input_currents_omit_activity_and_weight_backward_ops():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    @tf.function
    def gradients(spikes, weights):
        with tf.GradientTape() as tape:
            tape.watch((spikes, weights))
            currents = fused_spike_currents(
                spikes,
                weights,
                csr_weights,
                connectivity,
                basis,
                n_post=2,
                compute_spike_gradient=False,
                compute_weight_gradient=False,
            )
            loss = tf.reduce_sum(currents)
        spike_gradient, weight_gradient = tape.gradient(loss, (spikes, weights))
        return (
            tf.zeros_like(spikes) if spike_gradient is None else spike_gradient,
            tf.zeros_like(weights) if weight_gradient is None else weight_gradient,
        )

    concrete = gradients.get_concrete_function(
        tf.TensorSpec([2, 3], tf.float32),
        tf.TensorSpec([4], tf.float32),
    )
    graph = concrete.graph.as_graph_def()
    operation_types = {node.op for node in graph.node}
    for function in graph.library.function:
        operation_types.update(node.op for node in function.node_def)

    spike_gradient, weight_gradient = concrete(
        tf.ones([2, 3], tf.float32), master_weights
    )
    np.testing.assert_array_equal(spike_gradient, 0.0)
    np.testing.assert_array_equal(weight_gradient, 0.0)
    assert "DpointnetCsrSpikeGrad" not in operation_types
    assert "DpointnetCsrWeightGrad" not in operation_types


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize("activity_mode", ["zero", "dense", "poisson"])
def test_fused_sparse_input_batch32_matches_canonical_weight_gradient(
    activity_mode,
):
    rng = np.random.default_rng(59)
    n_pre = 100
    n_post = 11
    edges_per_source = 257
    pre_ids = np.repeat(np.arange(8), edges_per_source)
    post_ids = rng.integers(0, n_post, size=pre_ids.size)
    indices = np.column_stack((post_ids, pre_ids))
    synapse_types = rng.integers(0, 5, size=pre_ids.size)
    connectivity = build_csr_connectivity(
        indices,
        synapse_types,
        n_pre,
        n_post,
        5,
        build_compact_pairs=True,
    )
    master_weights = tf.Variable(
        rng.uniform(0.1, 1.0, size=pre_ids.size), dtype=tf.float32
    )
    csr_weights = reorder_csr_values(tf.cast(master_weights, tf.float16), connectivity)
    basis = tf.constant(rng.uniform(0.1, 1.0, size=(5, 4)), tf.float16)
    if activity_mode == "zero":
        activity = np.zeros((32, n_pre))
    elif activity_mode == "dense":
        activity = np.ones((32, n_pre))
    else:
        activity = rng.poisson(0.25, size=(32, n_pre))
    spike_counts = tf.constant(activity, dtype=tf.float16)
    loss_weights = tf.constant(
        rng.uniform(-0.5, 0.5, size=(32 * n_post, 4)), tf.float16
    )

    with tf.GradientTape() as tape:
        currents = fused_spike_currents(
            spike_counts,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=n_post,
            compute_spike_gradient=False,
            use_packed_sm120_backward="auto",
        )
        loss = tf.reduce_sum(currents * loss_weights)
    actual_gradient = tape.gradient(loss, master_weights)

    with tf.GradientTape() as tape:
        reference = _reference_currents_for_connectivity(
            spike_counts,
            master_weights,
            basis,
            indices,
            synapse_types,
            n_post,
        )
        reference_loss = tf.reduce_sum(reference * loss_weights)
    reference_gradient = tape.gradient(reference_loss, master_weights)

    np.testing.assert_allclose(
        actual_gradient, reference_gradient, rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_currents_execute_inside_tf_function():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    @tf.function
    def run(spikes):
        return fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )

    result = run(tf.constant([[1.0, 0.0, 2.0]], tf.float32))
    reference = _reference_currents(
        tf.constant([[1.0, 0.0, 2.0]], tf.float32),
        master_weights,
        basis,
    )
    np.testing.assert_allclose(result.numpy(), reference.numpy())


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_currents_dynamic_batch32_uses_correct_forward_path():
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 3, 2, 2, build_compact_pairs=True
    )
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, tf.float16), connectivity)
    basis = tf.constant([[1.0, 0.5, 0.25, 0.125], [0.25, 2.0, 0.75, 1.5]], tf.float16)

    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float16)])
    def run(spikes):
        return fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )

    spikes = tf.ones([32, 3], tf.float16)
    result = run(spikes)
    reference = _reference_currents(spikes, master_weights, basis)

    np.testing.assert_allclose(result, reference, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize("batch_size", [31, 32])
def test_fused_currents_ignore_negative_activity_for_all_forward_paths(batch_size):
    connectivity = build_csr_connectivity(
        INDICES, SYNAPSE_TYPES, 3, 2, 2, build_compact_pairs=True
    )
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(tf.cast(master_weights, tf.float16), connectivity)
    basis = tf.constant([[1.0, 0.5, 0.25, 0.125], [0.25, 2.0, 0.75, 1.5]], tf.float16)
    spikes = -tf.ones([batch_size, 3], tf.float16)

    currents = fused_spike_currents(
        spikes,
        master_weights,
        csr_weights,
        connectivity,
        basis,
        n_post=2,
        compute_spike_gradient=True,
    )

    np.testing.assert_array_equal(currents, 0.0)


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_currents_support_int64_connectivity():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    metadata = _metadata_values(connectivity).numpy().astype(np.int64)
    with tf.device("/GPU:0"):
        metadata_handle = csr_spike_ops._create_metadata_resource(metadata, tf.int64)
    connectivity = csr_spike_ops.CsrConnectivity(
        {
            **connectivity,
            "metadata_handle": metadata_handle,
            "index_dtype": "int64",
        }
    )
    master_weights = tf.Variable([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)
    spikes = tf.constant([[1.0, 0.0, 2.0]], dtype=tf.float32)

    result = fused_spike_currents(
        spikes,
        master_weights,
        csr_weights,
        connectivity,
        basis,
        n_post=2,
        compute_spike_gradient=True,
    )

    np.testing.assert_allclose(
        result.numpy(),
        _reference_currents(spikes, master_weights, basis).numpy(),
    )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_currents_support_empty_connectivity_and_gradients():
    connectivity = build_csr_connectivity(
        np.empty((0, 2), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        3,
        2,
        2,
    )
    master_weights = tf.Variable([], dtype=tf.float32)
    csr_weights = reorder_csr_values(master_weights, connectivity)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)
    spikes = tf.Variable([[1.0, 0.0, 2.0]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        currents = fused_spike_currents(
            spikes,
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )
        loss = tf.reduce_sum(currents)
    spike_gradient, weight_gradient = tape.gradient(loss, [spikes, master_weights])

    np.testing.assert_array_equal(currents.numpy(), np.zeros((2, 2)))
    np.testing.assert_array_equal(spike_gradient.numpy(), np.zeros((1, 3)))
    assert weight_gradient.shape == (0,)


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_currents_reject_mismatched_master_shape():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    master_weights = tf.Variable([1.0], dtype=tf.float32)
    csr_weights = tf.ones([4], dtype=tf.float32)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    with pytest.raises(ValueError, match="master_weights"):
        fused_spike_currents(
            tf.ones([1, 3]),
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
def test_fused_currents_reject_dynamic_mismatched_master_shape():
    connectivity = build_csr_connectivity(INDICES, SYNAPSE_TYPES, 3, 2, 2)
    csr_weights = tf.ones([4], dtype=tf.float32)
    basis = tf.constant([[1.0, 0.5], [0.25, 2.0]], dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def run(master_weights):
        return fused_spike_currents(
            tf.ones([1, 3]),
            master_weights,
            csr_weights,
            connectivity,
            basis,
            n_post=2,
            compute_spike_gradient=True,
        )

    with pytest.raises(tf.errors.InvalidArgumentError, match="master_weights"):
        run(tf.ones([1], dtype=tf.float32))
