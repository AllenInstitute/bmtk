import copy
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet import acceleration


@pytest.fixture
def precision_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


@pytest.fixture
def hardware(monkeypatch):
    csr = acceleration.csr_spike_ops
    state = acceleration.glif_state_ops
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 86)
    monkeypatch.setattr(csr, "fused_cuda_available", lambda: True)
    monkeypatch.setattr(csr, "fused_recurrent_accumulation_available", lambda: True)
    monkeypatch.setattr(csr, "cuda_op_status", lambda: "test CSR library")
    monkeypatch.setattr(state, "glif_state_op_status", lambda: "test state library")
    for name in (
        "fused_glif_state_available", "fused_nest_state_available",
        "fused_pascal_nest_launch_available",
        "fused_nest_type_indexed_coefficients_available",
        "fused_nest_event_vjp_available", "fused_nest_state_history_available",
        "fused_nest_state_history_type_indexed_available",
        "fused_voltage_penalty_available",
    ):
        monkeypatch.setattr(state, name, lambda: True)
    return csr, state


def resolve(params=None, **kwargs):
    args = dict(
        compute_dtype=tf.float16, variable_dtype=tf.float32,
        batch_size=32, basis_width=4,
        train_recurrent_per_type=False,
    )
    args.update(kwargs)
    return acceleration.resolve_acceleration_options(
        {"acceleration_profile": "auto", **(params or {})}, **args
    )


@pytest.mark.parametrize("architecture", [61, 70, 75, 80, 86, 89, 90, None])
@pytest.mark.parametrize("temporal", ["compute", "float32"])
def test_architecture_and_precision_gate_packed_flags(hardware, monkeypatch, architecture, temporal):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: architecture)
    params = {"temporal_gradient_precision": temporal}
    actual, report = resolve(params)
    enabled = architecture is not None and architecture >= 70 and temporal == "compute"
    assert actual["use_packed_sm120_backward"] == ("auto" if enabled else False)
    assert actual["use_packed_sm120_external_backward"] == ("auto" if enabled else False)
    assert report["architecture"] == architecture
    assert actual["temporal_gradient_precision"] == temporal
    assert actual["use_direct_state_rnn_loop"] is True


@pytest.mark.parametrize("batch", [1, 7, 8, 13, 32, 33, None])
@pytest.mark.parametrize("basis", [4, 5])
def test_batch_and_basis_admission(hardware, batch, basis):
    actual, _ = resolve(batch_size=batch, basis_width=basis)
    small_four_basis = batch is not None and batch <= 32 and basis == 4
    assert actual["use_device_active_queue_forward"] == small_four_basis
    assert actual["use_active_row_forward"] == small_four_basis
    assert actual["use_fixed4_input_forward"] == (basis == 4)
    assert actual["use_fused_state"] == (basis == 4)
    assert actual["use_packed_sm120_backward"] == (
        "auto" if batch == 32 and basis == 4 else False
    )


@pytest.mark.parametrize("dtype,masters", [
    (tf.float16, tf.float32), (tf.float32, tf.float32),
    (tf.bfloat16, tf.float32), (tf.float64, tf.float64),
])
def test_numeric_policy_not_changed(hardware, dtype, masters):
    actual, _ = resolve(compute_dtype=dtype, variable_dtype=masters)
    assert actual["use_fused_cuda"] == (dtype in (tf.float16, tf.float32))
    assert actual["use_packed_sm120_backward"] == ("auto" if dtype == tf.float16 else False)


@pytest.mark.parametrize("safe_binary", [False, True])
def test_pascal_nest_requires_occupancy_aware_state_binary(hardware, monkeypatch, safe_binary):
    csr, state = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 61)
    monkeypatch.setattr(state, "fused_pascal_nest_launch_available", lambda: safe_binary)
    actual, report = resolve({"dynamics_mode": "nest"})
    assert actual["dynamics_mode"] == "nest"
    assert actual["use_fused_cuda"] is True
    assert actual["use_fused_state"] is safe_binary
    assert actual["use_direct_state_rnn_loop"] is safe_binary
    assert actual["use_fused_recurrent_accumulation"] is safe_binary
    assert actual["use_packed_sm120_backward"] is False
    assert "occupancy" in report["reasons"]["use_fused_state"]


@pytest.mark.parametrize("profile", [None, "auto"])
@pytest.mark.parametrize("safe_binary", [False, True])
def test_explicit_pascal_state_requires_safe_launches(
    hardware, monkeypatch, precision_policy, profile, safe_binary
):
    from bmtk.simulator.dpointnet.cell_models import glif3_cell
    from test_precision_credit import make_cell

    csr, state = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 61)
    monkeypatch.setattr(state, "fused_pascal_nest_launch_available", lambda: safe_binary)
    monkeypatch.setattr(glif3_cell, "_gpu_compute_architecture", lambda: 61)
    monkeypatch.setattr(glif3_cell, "fused_nest_state_available", lambda: True)
    monkeypatch.setattr(glif3_cell, "fused_pascal_nest_launch_available", lambda: safe_binary)
    network, inputs, args = make_cell(mode="nest", return_spec=True)
    args.update(acceleration_profile=profile, use_fused_state=True, use_fused_cuda=False)
    if safe_binary:
        assert glif3_cell._resolve_fused_state(True, 4, True, "nest") is True
    else:
        with pytest.raises(ValueError, match="occupancy-aware"):
            glif3_cell.GLIF3Cell(network, inputs, **args)


@pytest.mark.parametrize("probe,option", [
    ("fused_nest_event_vjp_available", "use_fused_nest_event_vjp"),
    ("fused_nest_state_history_available", "use_fused_state_history"),
    ("fused_voltage_penalty_available", "use_native_voltage_penalty"),
    ("fused_nest_state_history_type_indexed_available", "use_type_indexed_nest_coefficients"),
])
def test_partial_state_library_falls_back_per_optional_operator(
    hardware, monkeypatch, probe, option
):
    _, state = hardware
    monkeypatch.setattr(state, probe, lambda: False)
    selected, report = resolve()
    assert selected["use_fused_state"] is True
    assert selected[option] is False
    assert report["selected"][option] is False
    explicit, _ = resolve({option: True})
    assert explicit[option] is True


def test_type_indexed_state_without_history_keeps_compatible_route(hardware, monkeypatch):
    _, state = hardware
    monkeypatch.setattr(state, "fused_nest_state_history_type_indexed_available", lambda: False)
    actual, _ = resolve({"use_fused_state_history": False})
    assert actual["use_type_indexed_nest_coefficients"] is True


@pytest.mark.parametrize("compute_dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("temporal", ["compute", "float32"])
@pytest.mark.parametrize("batch", [1, 8, 16, 32, 33])
def test_pascal_compatible_accumulator_not_packed_policy(hardware, monkeypatch, compute_dtype, temporal, batch):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 61)
    actual, _ = resolve(
        {"temporal_gradient_precision": temporal},
        compute_dtype=compute_dtype, batch_size=batch,
    )
    assert actual["use_fused_cuda"] is True
    assert actual["use_fused_state"] is True
    assert actual["use_fused_recurrent_accumulation"] == (batch <= 32)
    assert actual["use_javier_recurrent_vjp"] == (
        batch <= 32 and compute_dtype == tf.float16 and temporal == "compute"
    )
    assert actual["use_packed_sm120_backward"] is False
    assert actual["use_packed_sm120_external_backward"] is False


def test_missing_libraries_and_cpu_are_reported(hardware, monkeypatch):
    csr, state = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: None)
    monkeypatch.setattr(csr, "fused_cuda_available", lambda: False)
    monkeypatch.setattr(state, "fused_glif_state_available", lambda: False)
    monkeypatch.setattr(state, "fused_nest_state_available", lambda: False)
    actual, report = resolve()
    assert not any(actual[key] for key in report["selected"])
    assert report["csr_status"] == "test CSR library"


def test_explicit_overrides_and_scientific_options_preserved(hardware):
    requested = {
        "use_fused_cuda": False, "use_packed_sm120_backward": False,
        "dynamics_mode": "nest", "hard_reset": False, "detach_reset": True,
        "detach_asc_reset": False, "noise_seed": 1234,
        "temporal_gradient_precision": "compute", "state_precision": "compute",
        "recurrent_dampening_factor": 1.0, "lr_scale": 0.7,
    }
    retained = copy.deepcopy(requested)
    actual, report = resolve(requested)
    assert requested == retained
    for name, value in requested.items():
        assert actual[name] == value
    assert not actual["use_fused_current_accumulation"]
    assert not actual["use_device_active_queue_forward"]
    assert report["reasons"]["use_fused_cuda"].startswith("explicit override")


@pytest.mark.parametrize("carry,trainable,per_type,direct", [
    (False, True, False, True), (True, True, False, True),
    (True, False, False, True), (True, True, True, True),
    (True, True, False, False),
])
def test_accumulator_requires_exact_execution_route(hardware, carry, trainable, per_type, direct):
    actual, _ = resolve({
        "use_direct_state_rnn_loop": carry,
        "train_recurrent": trainable, "train_recurrent_per_type": per_type,
        "use_direct_csr_recurrent_gradient": direct,
    })
    enabled = carry and trainable and not per_type and direct
    assert actual["use_fused_recurrent_accumulation"] == enabled
    assert actual["use_javier_recurrent_vjp"] == enabled


def test_explicit_none_profile_is_exact_noop():
    requested = {"use_fused_cuda": False, "hard_reset": True, "acceleration_profile": None}
    actual, report = acceleration.resolve_acceleration_options(
        requested, compute_dtype=tf.float32, variable_dtype=tf.float32,
        batch_size=32, basis_width=4,
    )
    assert actual == {"use_fused_cuda": False, "hard_reset": True}
    assert actual is not requested
    assert report is None


def test_omitted_profile_selects_automatic_execution(hardware):
    actual, report = acceleration.resolve_acceleration_options(
        {}, compute_dtype=tf.float16, variable_dtype=tf.float32,
        batch_size=32, basis_width=4,
    )
    assert report["profile"] == "auto"
    assert actual["use_direct_state_rnn_loop"] is True
    assert actual["use_fused_recurrent_accumulation"] is True


def test_direct_cell_auto_keeps_canonical_tape_gradients(hardware):
    actual, report = resolve(canonical_gradient_boundary=False)
    assert actual["use_direct_csr_recurrent_gradient"] is False
    assert actual["use_fused_recurrent_accumulation"] is False
    assert actual["use_javier_recurrent_vjp"] is False
    assert "canonical-gradient boundary" in report["reasons"]["use_direct_csr_recurrent_gradient"]


def test_auto_rechecks_mutable_type_coefficients(hardware):
    actual, _ = resolve()
    assert actual["use_type_indexed_nest_coefficients"] is True
    assert actual["use_static_type_indexed_nest_dispatch"] is False


def test_standalone_per_type_default_does_not_admit_per_edge_accumulator(hardware):
    actual, _ = resolve(
        {"use_direct_state_rnn_loop": True}, train_recurrent_per_type=True
    )
    assert not actual["use_fused_recurrent_accumulation"]
    assert not actual["use_direct_csr_recurrent_gradient"]


@pytest.mark.parametrize("trainable,per_type", [(False, False), (True, True), (True, False)])
@pytest.mark.parametrize("explicit", [None, True, False])
def test_direct_csr_requires_per_edge_training_without_overriding_flags(hardware, trainable, per_type, explicit):
    params = {"train_recurrent": trainable, "train_recurrent_per_type": per_type}
    if explicit is not None:
        params["use_direct_csr_recurrent_gradient"] = explicit
    actual, report = resolve(params)
    assert actual["use_direct_csr_recurrent_gradient"] == (
        explicit if explicit is not None else trainable and not per_type
    )
    assert actual["train_recurrent"] == trainable
    assert actual["train_recurrent_per_type"] == per_type
    if explicit is not None:
        assert report["reasons"]["use_direct_csr_recurrent_gradient"].startswith("explicit override")


@pytest.mark.parametrize("flag", [
    "use_fixed4_input_forward", "use_active_row_forward", "use_device_active_queue_forward",
])
def test_five_basis_explicit_forward_flags_are_not_overridden(hardware, flag):
    actual, report = resolve({flag: True}, basis_width=5)
    assert actual[flag] is True
    assert report["reasons"][flag].startswith("explicit override")


@pytest.mark.parametrize("architecture", [61, 70, 75, 80, 86])
def test_native_carrier_rejects_incompatible_architecture_manifest(hardware, monkeypatch, architecture):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: architecture)
    monkeypatch.setattr(csr, "fused_cuda_available", lambda: False)
    # The accumulator entry point alone is not architecture qualification.
    assert csr.fused_recurrent_accumulation_available()
    with pytest.raises(ValueError, match="GPU/library"):
        acceleration.resolve_weight_carry_options({"use_fused_recurrent_accumulation": True})
    with pytest.raises(ValueError, match="compatible SM61"):
        acceleration.resolve_weight_carry_options({}, overrides={"native_accumulator": True})
    assert not acceleration.resolve_weight_carry_options({})["resolved"]["native_accumulator"]


@pytest.mark.parametrize("batch", [8, 16, 32])
def test_five_basis_auto_executes_general_projection(hardware, monkeypatch, batch):
    csr, _ = hardware
    actual, _ = resolve(basis_width=5, batch_size=batch)
    observed = []

    def forward(spikes, masters, metadata, weights, basis, scale, active_rows,
                incoming_pre, incoming_edges, incoming_types, initial, **kwargs):
        observed.append(kwargs)
        assert not kwargs["use_fixed4_forward"]
        assert not kwargs["use_device_active_queue_forward"]
        assert not kwargs["use_grouped_batch32_forward"]
        return tf.matmul(spikes, weights[:, None] * basis)

    monkeypatch.setattr(csr, "_OPS", SimpleNamespace(dpointnet_csr_spike_forward=forward))
    connectivity = {
        "n_edges": 4, "n_post": 1, "n_synapse_types": 4, "fixed4_incoming": True,
        "index_dtype": "uint32", "n_pairs": 4, "metadata_handle": None,
        "incoming_pre_ids": tf.range(4), "incoming_edge_ids": tf.range(4),
        "incoming_types": tf.range(4),
    }
    spikes = tf.constant(np.tile([0., 1., 1., 0.], (batch, 1)), tf.float32)
    weights = tf.constant([.25, .5, .75, 1.])
    basis = tf.constant(np.arange(20).reshape(4, 5) / 32, tf.float32)
    current = csr.fused_spike_currents(
        spikes, weights, weights, connectivity, basis, 1, compute_spike_gradient=True,
        use_fixed4_forward=actual["use_fixed4_input_forward"],
        use_active_row_forward=actual["use_active_row_forward"],
        use_device_active_queue_forward=actual["use_device_active_queue_forward"],
        use_packed_sm120_backward=actual["use_packed_sm120_backward"],
    )
    expected = np.einsum("be,e,ek->bk", spikes.numpy(), weights.numpy(), basis.numpy())
    np.testing.assert_array_equal(current, expected)
    assert len(observed) == 1


@pytest.mark.skipif(
    not acceleration.csr_spike_ops.fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
@pytest.mark.parametrize("batch", [8, 16, 32])
def test_actual_five_basis_auto_projection_and_gradients(batch):
    csr = acceleration.csr_spike_ops
    options, _ = resolve(batch_size=batch, basis_width=5, compute_dtype=tf.float32)
    indices = np.array([[0, pre] for pre in range(4)], dtype=np.int64)
    types = np.arange(4, dtype=np.int64)
    conn = csr.build_csr_connectivity(
        indices, types, 4, 1, 4, build_compact_pairs=True, build_fixed4_incoming=True
    )
    try:
        assert conn["fixed4_incoming"]
        spikes = tf.constant(np.tile([0., 1., 1., 0.], (batch, 1)), tf.float32)
        weights = tf.constant([.25, .5, .75, 1.])
        basis = tf.constant(np.arange(20).reshape(4, 5) / 32, tf.float32)
        shadow = csr.reorder_csr_values(weights, conn)
        with tf.GradientTape() as tape:
            tape.watch((spikes, weights))
            current = csr.fused_spike_currents(
                spikes, weights, shadow, conn, basis, 1, compute_spike_gradient=True,
                use_fixed4_forward=options["use_fixed4_input_forward"],
                use_active_row_forward=options["use_active_row_forward"],
                use_device_active_queue_forward=options["use_device_active_queue_forward"],
                use_packed_sm120_backward=options["use_packed_sm120_backward"],
            )
            loss = tf.reduce_sum(current)
        dz, dw = tape.gradient(loss, (spikes, weights))
        expected = np.einsum("be,e,ek->bk", spikes.numpy(), weights.numpy(), basis.numpy())
        np.testing.assert_array_equal(current, expected)
        np.testing.assert_array_equal(dw, spikes.numpy().sum(0) * basis.numpy().sum(1))
        np.testing.assert_array_equal(
            dz, np.broadcast_to(weights.numpy() * basis.numpy().sum(1), (batch, 4))
        )
    finally:
        conn.close()


@pytest.mark.parametrize("batch", [1, 5, 7, 8, 16, 32])
@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("pair", ["auto", True, False])
def test_pair_auto_inherits_current_profile_policy(hardware, batch, mode, dtype, pair):
    params = {"dynamics_mode": mode, "use_direct_state_rnn_loop": True}
    omitted, _ = resolve(params, batch_size=batch, compute_dtype=dtype)
    actual, report = resolve(
        {**params, "use_pair_projection": pair}, batch_size=batch, compute_dtype=dtype,
    )
    assert actual["use_pair_projection"] is (pair is not False)
    for name in ("use_fused_recurrent_accumulation", "use_javier_recurrent_vjp"):
        assert actual[name] == (omitted[name] if pair is not False else False)
    assert report["requested"]["use_pair_projection"] == pair


def test_compatibility_v1_preserves_historical_batch32_pair_heuristic(hardware):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import _resolve_pair_projection

    actual, report = resolve({
        "acceleration_profile": "compatibility_v1", "use_pair_projection": "auto",
    }, batch_size=7)
    assert actual["use_pair_projection"] == "auto"
    assert report is None
    assert not _resolve_pair_projection(actual["use_pair_projection"], True, 7, 4)
    assert _resolve_pair_projection(actual["use_pair_projection"], True, 32, 4)


def test_null_profile_warns_for_historical_auto():
    with pytest.warns(FutureWarning, match="compatibility_v1"):
        actual, report = resolve({"acceleration_profile": None, "use_pair_projection": "auto"})
    assert actual["use_pair_projection"] == "auto"
    assert report is None


def test_all_profile_managed_auto_requests_match_omission(hardware):
    from bmtk.simulator.dpointnet.migration import PROFILE_EXECUTION_OPTIONS

    omitted, report = resolve()
    assert set(report["selected"]) == set(PROFILE_EXECUTION_OPTIONS)
    params = {name: "auto" for name in report["selected"] if name.startswith("use_")}
    actual, explicit = resolve(params)
    assert actual == omitted
    assert explicit["provenance"]["use_pair_projection"] == "explicit_auto"
    assert report["provenance"]["use_pair_projection"] == "profile"


def test_pair_policy_defers_unknown_basis_width_to_constructor(hardware):
    actual, _ = resolve({"use_pair_projection": "auto"}, basis_width=None, batch_size=5)
    assert actual["use_pair_projection"] is True
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import _resolve_pair_projection
    with pytest.raises(ValueError, match="basis must have positive width"):
        _resolve_pair_projection(actual["use_pair_projection"], True, 5, 0)


def test_final_topology_fallback_is_not_an_avoidable_override(hardware):
    selected, report = resolve()
    selected = dict(report["selected"])
    selected["use_fixed4_input_forward"] = False
    final = acceleration.finalize_acceleration_report(
        report, report["requested"], selected,
        predicates={"use_fixed4_input_forward": {"exactly four incoming edges": False}},
    )
    entry = final["effective"]["use_fixed4_input_forward"]
    assert entry["provenance"] == "profile"
    assert entry["blocked_by"] == ["exactly four incoming edges"]
    audit = next(item for item in final["optimization_audit"] if item["option"] == "use_fixed4_input_forward")
    assert audit["classification"] == "eligibility_or_dependency"


def test_explicit_enabled_choice_is_not_reported_blocked_by_auto_policy(hardware):
    _, report = resolve({"use_static_type_indexed_nest_dispatch": True})
    final = acceleration.finalize_acceleration_report(
        report, report["requested"], report["selected"],
    )
    assert final["effective"]["use_static_type_indexed_nest_dispatch"]["blocked_by"] == []


def test_resolver_snapshot_survives_constructor_fallback_and_refinalization(hardware):
    _, report = resolve(batch_size=5)
    name = "use_fused_recurrent_accumulation"
    assert report["selected"][name] is True
    report["selected"][name] = False
    for _ in range(2):
        final = acceleration.finalize_acceleration_report(
            report, report["requested"], report["selected"],
            predicates={name: {"nonempty compact-pair connectivity": False}},
        )
        assert final["resolver_selected"][name] is True
        assert final["selected"][name] is False


@pytest.mark.parametrize("name,predicate", [
    ("use_fixed4_input_forward", "exactly four incoming edges"),
    ("use_fused_recurrent_accumulation", "nonempty compact-pair connectivity"),
    ("use_fused_recurrent_accumulation", "uint32 recurrent connectivity"),
])
@pytest.mark.parametrize("eligible", [False, True])
def test_override_audit_requires_final_independent_eligibility(hardware, name, predicate, eligible):
    _, report = resolve({name: False}, batch_size=5)
    assert report["automatic"][name] is True
    final = acceleration.finalize_acceleration_report(
        report, report["requested"], report["selected"],
        predicates={name: {predicate: eligible}},
    )
    entry = final["effective"][name]
    assert entry["resolved"] is False
    assert entry["current_policy"] is eligible
    audit = next(item for item in final["optimization_audit"] if item["option"] == name)
    assert audit["classification"] == (
        "avoidable_execution_override" if eligible else "eligibility_or_dependency"
    )


@pytest.mark.skipif(
    not acceleration.csr_spike_ops.fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
@pytest.mark.parametrize("counts", [(1, 1), (4, 4), (2, 6)])
@pytest.mark.parametrize("option", ["auto", False])
def test_native_audit_uses_topology_not_fixed4_override(precision_policy, counts, option):
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    network, inputs, args = make_cell(
        return_spec=True, fused=True, acceleration_profile="auto",
        batch_size=5, use_fixed4_input_forward=option,
    )
    net = inputs["drive"]
    net.update(
        input_type="spikes",
        indices=np.array([[post, 0] for post, count in enumerate(counts) for _ in range(count)]),
        weights=np.ones(sum(counts)), delays=np.ones(sum(counts)),
        syn_ids=np.zeros(sum(counts), dtype=np.int64),
    )
    cell = GLIF3Cell(network, inputs, **args)
    try:
        eligible = counts == (4, 4)
        entry = cell.acceleration_report["effective"]["use_fixed4_input_forward"]
        assert entry["current_policy"] is eligible
        assert entry["resolved"] is (eligible and option == "auto")
        assert cell.inputs["drive"]["fused_connectivity"]["fixed4_eligible"] is eligible
        audits = [
            item for item in cell.acceleration_report["optimization_audit"]
            if item["option"] == "use_fixed4_input_forward"
        ]
        assert any(item["classification"] == "avoidable_execution_override" for item in audits) is (
            eligible and option is False
        )
    finally:
        cell.close_fused_cuda()


@pytest.mark.skipif(
    not acceleration.csr_spike_ops.fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
@pytest.mark.parametrize("disabled", [False, True])
def test_native_empty_recurrence_preserves_resolver_snapshot_and_audit(precision_policy, disabled):
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    network, inputs, args = make_cell(
        return_spec=True, fused=True, acceleration_profile="auto",
        batch_size=5, _canonical_gradient_boundary=True,
        **({"use_fused_recurrent_accumulation": False} if disabled else {}),
    )
    for key in ("indices", "weights", "delays", "syn_ids"):
        network["synapses"][key] = network["synapses"][key][:0]
    cell = GLIF3Cell(network, inputs, **args)
    try:
        report = cell.acceleration_report
        name = "use_fused_recurrent_accumulation"
        assert report["resolver_selected"][name] is (not disabled)
        entry = report["effective"][name]
        assert entry["resolved"] is False
        assert entry["current_policy"] is False
        assert "nonempty compact-pair connectivity" in entry["blocked_by"]
        assert not any(
            item["classification"] == "avoidable_execution_override" and item["option"] == name
            for item in report["optimization_audit"]
        )
    finally:
        cell.close_fused_cuda()


@pytest.mark.skipif(
    not acceleration.csr_spike_ops.fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
def test_native_packed_input_override_does_not_hide_metadata_eligibility(precision_policy):
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    if not acceleration.csr_spike_ops._auto_native_architecture(
        acceleration.csr_spike_ops._gpu_compute_architecture(),
    ):
        pytest.skip("Automatic packed backward requires SM70+")
    network, inputs, args = make_cell(
        return_spec=True, fused=True, acceleration_profile="auto",
        batch_size=32, use_packed_sm120_external_backward=False,
    )
    inputs["drive"]["input_type"] = "spikes"
    inputs["current"] = copy.deepcopy(inputs["drive"])
    inputs["current"]["input_type"] = "current"
    cell = GLIF3Cell(network, inputs, **args)
    try:
        entry = cell.acceleration_report["effective"]["use_packed_sm120_external_backward"]
        assert entry["resolved"] is False
        assert entry["current_policy"] is True
        assert cell.inputs["drive"]["fused_connectivity"]["n_pairs"] == 0
        paths = cell.acceleration_report["input_paths"]
        assert paths["drive"]["packed_eligible"] is True
        assert paths["current"]["packed_eligible"] is False
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("architecture", [None, 61, 70, 75, 80, 86])
@pytest.mark.parametrize("available", [False, True])
def test_auto_pair_equivalence_preserves_hardware_guards(hardware, monkeypatch, architecture, available):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: architecture)
    monkeypatch.setattr(csr, "fused_cuda_available", lambda: available and architecture is not None)
    omitted, _ = resolve(batch_size=5)
    actual, _ = resolve({"use_pair_projection": "auto"}, batch_size=5)
    assert actual == omitted
    if not available or architecture is None:
        assert not actual["use_pair_projection"]
        assert not actual["use_fused_recurrent_accumulation"]
        assert not actual["use_javier_recurrent_vjp"]


def test_cpu_final_report_contains_no_unresolved_auto(precision_policy):
    from test_precision_credit import make_cell

    cell = make_cell(
        acceleration_profile="auto", use_pair_projection="auto",
        use_fused_cuda=False, use_fused_state=False, batch_size=5,
    )
    report = cell.acceleration_report
    assert report["stage"] == "cell"
    assert report["selected"]["use_pair_projection"] is False
    assert "auto" not in report["selected"].values()
    pair = report["effective"]["use_pair_projection"]
    assert pair["requested"] == "auto"
    assert pair["provenance"] == "explicit_auto"
    assert "compatible fused CUDA currents" in pair["blocked_by"]
    assert any(
        "use_pair_projection=True" in item["message"]
        for item in report["optimization_audit"]
    )


def test_final_report_distinguishes_scientific_incompatibility_and_override(hardware):
    options, report = resolve({"dynamics_mode": "legacy", "use_pair_projection": False}, batch_size=5)
    final = acceleration.finalize_acceleration_report(
        report, report["requested"], options,
    )
    entries = {item["option"]: item for item in final["optimization_audit"]}
    assert entries["use_pair_projection"]["classification"] == "avoidable_execution_override"
    assert entries["use_combined_current_sweep"]["classification"] == "scientific_incompatibility"
    assert "NEST dynamics" in final["effective"]["use_combined_current_sweep"]["blocked_by"]
    assert "use_pair_projection=True" in final["effective"]["use_fused_recurrent_accumulation"]["blocked_by"]


@pytest.mark.skipif(
    not acceleration.csr_spike_ops.fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
@pytest.mark.parametrize("profile", ["auto", "compatibility_v1"])
@pytest.mark.parametrize("pair", ["auto", False])
def test_native_batch5_reports_requested_policy_and_actual_pairing(precision_policy, profile, pair):
    from test_precision_credit import make_cell

    cell = make_cell(
        batch_size=5, acceleration_profile=profile, use_pair_projection=pair,
        fused=True, use_fused_cuda=True, _canonical_gradient_boundary=True,
    )
    try:
        report = cell.acceleration_report
        entry = report["effective"]["use_pair_projection"]
        assert entry["requested"] == pair
        assert entry["resolved"] is (profile == "auto" and pair == "auto")
        assert report["selected"]["use_pair_projection"] == cell._use_pair_projection
        if profile == "compatibility_v1" and pair == "auto":
            assert entry["provenance"] == "explicit_compatibility_override"
            assert entry["current_policy"] is True
            assert "compatibility_v1 auto requires batch32 and four bases" in entry["blocked_by"]
            assert any(
                "blocks fused recurrent accumulation and Javier VJP" in item["message"]
                for item in report["optimization_audit"]
            )
    finally:
        cell.close_fused_cuda()


@pytest.mark.skipif(
    not acceleration.csr_spike_ops.fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
@pytest.mark.parametrize("batch", [1, 5, 7, 8, 16, 32])
@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("selective", [False, True])
def test_native_omitted_and_auto_pair_values_vjps_and_restore(
    precision_policy, tmp_path, batch, mode, selective,
):
    from test_precision_credit import make_cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    csr = acceleration.csr_spike_ops
    if not csr._auto_native_architecture(csr._gpu_compute_architecture()):
        pytest.skip("Automatic accumulator requires SM70+")
    cells = [
        make_cell(
            mode=mode, selective=selective, fused=True, acceleration_profile="auto",
            batch_size=batch, _canonical_gradient_boundary=True,
            **({"use_pair_projection": "auto"} if explicit else {}),
        )
        for explicit in (False, True)
    ]
    results = []
    try:
        for index, cell in enumerate(cells):
            assert cell._use_pair_projection is True
            assert cell.use_fused_recurrent_accumulation is True
            assert cell.use_javier_recurrent_vjp is selective
            assert cell.acceleration_report["selected"]["use_pair_projection"] is True
            layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
            states = list(cell.zero_state(batch, cell.compute_dtype))
            states[0] = tf.ones_like(states[0])
            sequences = tf.ones([batch, 5, 2], cell.compute_dtype)

            @tf.function
            def evaluate():
                with tf.GradientTape() as tape:
                    output = layer(sequences, initial_state=states)
                    loss = sum(
                        tf.reduce_sum(tf.cast(value, tf.float32))
                        for value in tf.nest.flatten(output)
                        if value.dtype.is_floating
                    )
                return output, tape.gradient(loss, cell.recurrent_weight_values)

            output, gradient = evaluate()
            assert gradient is not None
            assert np.all(np.isfinite(gradient))
            assert np.any(gradient.numpy() != 0)
            results.append((output, gradient))
            original = cell.canonical_weight_values().numpy().copy()
            checkpoint = tf.train.Checkpoint(cell=cell)
            path = checkpoint.save(str(tmp_path / f"cell-{index}"))
            cell.recurrent_weight_values.assign_add(tf.ones_like(cell.recurrent_weight_values))
            checkpoint.restore(path).assert_consumed()
            cell.refresh_recurrent_weight_shadow()
            np.testing.assert_array_equal(cell.canonical_weight_values(), original)
            replay, replay_gradient = evaluate()
            for before, after in zip(tf.nest.flatten(output), tf.nest.flatten(replay)):
                np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(gradient, replay_gradient)
            assert cell.acceleration_report["rollout_selected"]["use_direct_state_rnn_loop"]
        for left, right in zip(tf.nest.flatten(results[0]), tf.nest.flatten(results[1])):
            np.testing.assert_array_equal(left, right)
    finally:
        for cell in cells:
            cell.close_fused_cuda()


@pytest.mark.parametrize("profile", [True, False, "fast", 1])
def test_invalid_profile_rejected(profile):
    with pytest.raises(ValueError, match="acceleration_profile"):
        resolve({"acceleration_profile": profile})


@pytest.mark.parametrize("architecture", [70, 75, 80, 86, 89])
def test_runner_carriers_gate_native_op_independently(hardware, monkeypatch, architecture):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: architecture)
    monkeypatch.setattr(csr, "weight_only_accumulation_available", lambda: True)
    options, _ = resolve({"use_direct_state_rnn_loop": True})
    rec = acceleration.resolve_weight_carry_options(options)
    audio = acceleration.resolve_weight_carry_options(options, stopped_input=True)
    eligible = architecture >= 70
    assert rec["resolved"]["native_accumulator"] == eligible
    assert rec["resolved"]["compute_spike_gradient"] is True
    assert audio["resolved"]["native_accumulator"] == eligible
    assert audio["resolved"]["compute_spike_gradient"] == (not eligible)
    assert options["use_packed_sm120_external_backward"] == ("auto" if eligible else False)


def test_incompatible_explicit_carrier_override_rejected(hardware, monkeypatch):
    csr, _ = hardware
    # SM75 admits the automatic accumulator only with FP16 temporal backward.
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 75)
    options, _ = resolve({"use_direct_state_rnn_loop": True}, compute_dtype=tf.float32)
    with pytest.raises(ValueError, match="admitted carrier route"):
        acceleration.resolve_weight_carry_options(options, overrides={"native_accumulator": True})
    with pytest.raises(ValueError, match="stopped"):
        acceleration.resolve_weight_carry_options(options, overrides={"compute_spike_gradient": False})
    with pytest.raises(ValueError, match="SM61/SM70"):
        resolve({"use_direct_state_rnn_loop": True, "use_fused_recurrent_accumulation": True},
                compute_dtype=tf.float32)
    with pytest.raises(ValueError, match="FP16"):
        resolve({"use_direct_state_rnn_loop": True, "use_javier_recurrent_vjp": True},
                compute_dtype=tf.float32)


@pytest.mark.parametrize("architecture", [70, 80])
@pytest.mark.parametrize("batch", [8, 32])
def test_volta_ampere_auto_admit_native_accumulator(hardware, monkeypatch, architecture, batch):
    # Admitted with the joint sweep on every SM70+ fat-binary target.
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: architecture)
    actual, report = resolve({"use_direct_state_rnn_loop": True}, batch_size=batch)
    assert actual["use_fused_recurrent_accumulation"] is True
    assert actual["use_javier_recurrent_vjp"] is True
    assert actual["use_packed_sm120_backward"] == ("auto" if batch == 32 else False)
    assert actual["use_packed_sm120_external_backward"] == ("auto" if batch == 32 else False)
    assert "SM70+" in report["reasons"]["use_fused_recurrent_accumulation"]
    explicit, _ = resolve({"use_direct_state_rnn_loop": True,
                           "use_fused_recurrent_accumulation": True}, batch_size=batch)
    assert explicit["use_fused_recurrent_accumulation"] is True


@pytest.mark.parametrize("batch", [8, 16, 32])
def test_rtx8000_auto_keeps_batch_specific_guards(hardware, monkeypatch, batch):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 75)
    actual, _ = resolve({"use_direct_state_rnn_loop": True}, batch_size=batch)
    assert actual["use_fused_recurrent_accumulation"] is True
    assert actual["use_javier_recurrent_vjp"] is True
    assert actual["use_packed_sm120_backward"] == ("auto" if batch == 32 else False)
    assert actual["use_packed_sm120_external_backward"] == ("auto" if batch == 32 else False)


def test_pascal_carrier_uses_compatible_automatic_accumulator(hardware, monkeypatch):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 61)
    options, _ = resolve({"use_direct_state_rnn_loop": True})
    assert acceleration.resolve_weight_carry_options(options)["resolved"]["native_accumulator"] is True
    explicit = {"use_fused_recurrent_accumulation": True, "use_javier_recurrent_vjp": True}
    assert acceleration.resolve_weight_carry_options(explicit)["resolved"]["native_accumulator"] is True


@pytest.mark.parametrize("compute,temporal", [
    (tf.float32, "compute"), (tf.float16, "float32"),
])
def test_rtx8000_auto_keeps_unqualified_precision_routes_generic(hardware, monkeypatch, compute, temporal):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 75)
    actual, _ = resolve(
        {"use_direct_state_rnn_loop": True, "temporal_gradient_precision": temporal},
        compute_dtype=compute,
    )
    assert actual["use_fused_recurrent_accumulation"] is False
    assert actual["use_javier_recurrent_vjp"] is False


def test_auto_accumulator_requires_four_bases(hardware):
    actual, _ = resolve({"use_direct_state_rnn_loop": True}, basis_width=5)
    assert actual["use_fused_recurrent_accumulation"] is False


def test_generic_carrier_keeps_live_credit_without_calling_accumulator(hardware, monkeypatch):
    import numpy as np
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 70)
    monkeypatch.setattr(csr, "fused_recurrent_weight_carry", lambda *a, **k: pytest.fail("native op called"))
    monkeypatch.setattr(csr, "fused_spike_currents", lambda z, carrier, *a, **k: z * tf.reduce_sum(carrier))
    options, _ = resolve({"use_direct_state_rnn_loop": True, "use_fused_recurrent_accumulation": False})
    for stopped in (False, True):
        route = acceleration.resolve_weight_carry_options(options, stopped_input=stopped)["resolved"]
        spikes, carrier = tf.ones((2, 2)), tf.ones((3,))
        with tf.GradientTape() as tape:
            tape.watch((spikes, carrier))
            current, identity = acceleration.project_weight_carry(
                spikes, carrier, None, None, None, 2, 1., resolved=route,
            )
            loss = tf.reduce_sum(current) + tf.reduce_sum(identity)
        dz, dw = tape.gradient(loss, (spikes, carrier))
        assert (dz is None) == stopped
        if not stopped:
            np.testing.assert_array_equal(dz, np.full((2, 2), 3.))
        np.testing.assert_array_equal(dw, np.full(3, 5.))


@pytest.mark.skipif(
    not acceleration.csr_spike_ops.fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
@pytest.mark.parametrize("stopped", [False, True])
def test_actual_auto_carrier_matches_independent_dense_oracle(stopped):
    import numpy as np
    csr = acceleration.csr_spike_ops
    indices = np.array([[0, 0], [1, 1], [1, 0]], dtype=np.int64)
    types = np.zeros(3, dtype=np.int64)
    conn = csr.build_csr_connectivity(indices, types, 2, 2, 1, build_compact_pairs=True)
    try:
        options, _ = resolve({"use_direct_state_rnn_loop": True})
        route = acceleration.resolve_weight_carry_options(options, stopped_input=stopped)["resolved"]
        # The generic route is the exact newly qualified SM70/75/80 path.
        if route["native_accumulator"]:
            pytest.skip("Native carrier oracle covered by weight-only accumulator tests")
        spikes = tf.constant([[0., 1.], [1., 0.]] * 16, tf.float16)
        weights = tf.constant([.125, .25, .5], tf.float32)
        basis = tf.constant([[.5, .25, .125, .0625]], tf.float16)
        carrier = csr.reorder_csr_values(weights, conn)
        shadow = tf.cast(carrier, tf.float16)
        with tf.GradientTape() as tape:
            tape.watch((spikes, carrier))
            current, identity = acceleration.project_weight_carry(
                spikes, carrier, shadow, conn, basis, 2, .5, resolved=route,
            )
            loss = tf.reduce_sum(tf.cast(current, tf.float32)) + tf.reduce_sum(identity)
        dz, dw = tape.gradient(loss, (spikes, carrier))
        tensor = np.zeros((2, 2, 4), dtype=np.float32)
        for edge, (post, pre) in enumerate(indices):
            tensor[pre, post] += weights.numpy()[edge] * basis.numpy()[0]
        expected = np.einsum("bp,pnk->bnk", spikes.numpy().astype(np.float32), tensor)
        np.testing.assert_allclose(current.numpy().reshape(32, 2, 4), expected, rtol=2e-5, atol=2e-6)
        expected_dw = spikes.numpy().astype(np.float32).sum(0)[indices[:, 1]] * basis.numpy().sum() + 1.
        np.testing.assert_allclose(csr.restore_csr_values(dw, conn), expected_dw, rtol=2e-5, atol=2e-6)
        if stopped:
            assert dz is None
        else:
            # Spike zero does not imply zero credit.
            np.testing.assert_allclose(dz.numpy(), np.broadcast_to(tensor.sum((1, 2)) * .5, (32, 2)), rtol=2e-5, atol=2e-6)
            assert np.all(dz.numpy()[spikes.numpy() == 0] != 0)
    finally:
        conn.close()


@pytest.mark.parametrize("stopped", [False, True])
def test_generic_carrier_nested_recompute_while_loop_resource_capture(hardware, monkeypatch, stopped):
    import numpy as np
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 70)
    monkeypatch.setattr(csr, "fused_recurrent_weight_carry", lambda *a, **k: pytest.fail("native op called"))
    metadata = tf.Variable(2., trainable=False)

    def projection(z, carrier, shadow, connectivity, *args, **kwargs):
        return z * tf.reduce_sum(carrier) * connectivity["metadata_handle"].read_value()

    monkeypatch.setattr(csr, "fused_spike_currents", projection)
    options, _ = resolve({"use_direct_state_rnn_loop": True, "use_fused_recurrent_accumulation": False})
    route = acceleration.resolve_weight_carry_options(options, stopped_input=stopped)["resolved"]

    @tf.recompute_grad
    def segment(z, carrier):
        def step(index, credit, loss):
            current, credit = acceleration.project_weight_carry(
                z, credit, None, {"metadata_handle": metadata}, None, 2, 1., resolved=route,
            )
            return index + 1, credit, loss + tf.reduce_sum(current)
        _, credit, loss = tf.while_loop(
            lambda index, *_: index < 3, step, (0, carrier, tf.constant(0.)),
        )
        return loss + tf.reduce_sum(credit)

    @tf.function
    def gradients(z, carrier):
        with tf.GradientTape() as tape:
            tape.watch((z, carrier))
            loss = segment(z, carrier)
        dz, dw = tape.gradient(loss, (z, carrier))
        return loss, dz, dw

    spikes = tf.constant([[0., 1.], [1., 0.]])
    loss, dz, dw = gradients(spikes, tf.ones((3,)))
    np.testing.assert_array_equal(loss, 39.)
    np.testing.assert_array_equal(dw, np.full(3, 13.))
    # recompute_grad materializes zeros for deliberately stopped inputs.
    np.testing.assert_array_equal(dz, np.zeros((2, 2)) if stopped else np.full((2, 2), 18.))
