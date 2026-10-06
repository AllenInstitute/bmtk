import copy

import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet import acceleration


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
        "fused_nest_type_indexed_coefficients_available",
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
    enabled = architecture is not None and architecture >= 86 and temporal == "compute"
    assert actual["use_packed_sm120_backward"] == ("auto" if enabled else False)
    assert actual["use_packed_sm120_external_backward"] == ("auto" if enabled else False)
    assert report["architecture"] == architecture
    assert actual["temporal_gradient_precision"] == temporal
    assert "use_direct_state_rnn_loop" not in actual


@pytest.mark.parametrize("batch", [1, 7, 8, 13, 32, 33, None])
@pytest.mark.parametrize("basis", [4, 5])
def test_batch_and_basis_admission(hardware, batch, basis):
    actual, _ = resolve(batch_size=batch, basis_width=basis)
    assert actual["use_device_active_queue_forward"] == (batch is not None and batch <= 32)
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


def test_pascal_nest_not_silently_promoted(hardware, monkeypatch):
    csr, _ = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: 61)
    actual, report = resolve({"dynamics_mode": "nest"})
    assert actual["dynamics_mode"] == "nest"
    assert actual["use_fused_cuda"] is False
    assert actual["use_fused_state"] is False
    assert "Pascal" in report["reasons"]["use_fused_cuda"]


def test_missing_libraries_and_cpu_are_reported(hardware, monkeypatch):
    csr, state = hardware
    monkeypatch.setattr(csr, "_gpu_compute_architecture", lambda: None)
    monkeypatch.setattr(csr, "fused_cuda_available", lambda: False)
    monkeypatch.setattr(state, "fused_glif_state_available", lambda: False)
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


def test_no_profile_is_exact_noop():
    requested = {"use_fused_cuda": False, "hard_reset": True}
    actual, report = acceleration.resolve_acceleration_options(
        requested, compute_dtype=tf.float32, variable_dtype=tf.float32,
        batch_size=32, basis_width=4,
    )
    assert actual == requested
    assert actual is not requested
    assert report is None


def test_standalone_per_type_default_does_not_admit_per_edge_accumulator(hardware):
    actual, _ = resolve(
        {"use_direct_state_rnn_loop": True}, train_recurrent_per_type=True
    )
    assert not actual["use_fused_recurrent_accumulation"]


def test_explicit_auto_pair_metadata_does_not_admit_variable_batch_accumulator(hardware):
    actual, _ = resolve(
        {"use_direct_state_rnn_loop": True, "use_pair_projection": "auto"},
        batch_size=7,
    )
    assert not actual["use_fused_recurrent_accumulation"]


@pytest.mark.parametrize("profile", [True, False, "fast", 1])
def test_invalid_profile_rejected(profile):
    with pytest.raises(ValueError, match="acceleration_profile"):
        resolve({"acceleration_profile": profile})
