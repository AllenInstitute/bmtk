"""Opt-in hardware-aware execution choices, independent of scientific settings."""

from numbers import Integral

import tensorflow as tf

from .custom_ops import csr_spike_ops, glif_state_ops
from .io_tools import io


def resolve_acceleration_options(
    cell_params, *, compute_dtype, variable_dtype, batch_size, basis_width,
    train_recurrent_per_type=True,
):
    """Resolve ``acceleration_profile="auto"`` without overriding explicit flags.

    The returned report is suitable for recording with execution provenance.
    Omitting the profile preserves existing constructor defaults.
    """
    options = dict(cell_params)
    profile = options.pop("acceleration_profile", None)
    if profile is None:
        return options, None
    if profile != "auto":
        raise ValueError('acceleration_profile must be None or "auto".')

    compute_dtype = tf.as_dtype(compute_dtype)
    variable_dtype = tf.as_dtype(variable_dtype)
    architecture = csr_spike_ops._gpu_compute_architecture()
    build_info = tf.sysconfig.get_build_info()
    report = {
        "profile": profile,
        "architecture": architecture,
        "tensorflow": tf.__version__,
        "tensorflow_cuda": build_info.get("cuda_version"),
        "tensorflow_cudnn": build_info.get("cudnn_version"),
        "csr_status": csr_spike_ops.cuda_op_status(),
        "state_status": glif_state_ops.glif_state_op_status(),
        "requested": dict(cell_params),
        "selected": {},
        "reasons": {},
    }

    def select(name, enabled, reason):
        if name in options:
            report["reasons"][name] = "explicit override; constructor validation retained"
        else:
            options[name] = bool(enabled)
            report["reasons"][name] = reason
        report["selected"][name] = options[name]

    numeric = compute_dtype in (tf.float16, tf.float32) and variable_dtype == tf.float32
    nest = options.get("dynamics_mode", "legacy") == "nest"
    pascal_nest = nest and architecture is not None and architecture < 70
    currents = numeric and csr_spike_ops.fused_cuda_available() and not pascal_nest
    select(
        "use_fused_cuda", currents,
        "compatible loaded CSR library and FP16/FP32 compute with FP32 masters; "
        "general NEST/Pascal and multiple visible GPUs excluded",
    )
    currents = currents and options["use_fused_cuda"] is not False
    known_batch = isinstance(batch_size, Integral) and not isinstance(batch_size, bool) and batch_size > 0
    small_batch = known_batch and batch_size <= 32
    four_basis = basis_width == 4
    fp16_backward = compute_dtype == tf.float16 and options.get(
        "temporal_gradient_precision", "compute"
    ) == "compute"
    state_available = (
        glif_state_ops.fused_nest_state_available()
        if nest else glif_state_ops.fused_glif_state_available()
    )
    select(
        "use_fused_state",
        numeric and four_basis and state_available and not pascal_nest,
        "compatible mode-specific state library, FP32 masters and four bases; "
        "general NEST/Pascal excluded",
    )
    state = options["use_fused_state"] is not False and numeric and four_basis and state_available and not pascal_nest
    select("use_pair_projection", currents and known_batch,
           "compatible CSR library and known positive batch")
    select("use_fixed4_input_forward", currents,
           "compatible CSR library; each input checks exactly four incoming edges")
    select("use_fused_current_accumulation", currents and four_basis and batch_size == 32,
           "compatible CSR library, four bases and batch32")
    select("use_direct_csr_recurrent_gradient", currents,
           "compatible CSR library; canonical master ordering retained")
    select("use_active_row_forward", currents and small_batch,
           "compatible CSR library and batch1..32")
    select("use_device_active_queue_forward", currents and small_batch,
           "compatible CSR library and batch1..32")
    pair_projection = options["use_pair_projection"] is True or (
        options["use_pair_projection"] == "auto" and batch_size == 32 and four_basis
    )
    packed = (
        currents and fp16_backward and four_basis and batch_size == 32
        and architecture is not None and architecture >= 86 and pair_projection
    )
    select("use_packed_sm120_backward", packed,
           "FP16 temporal backward, SM86+, batch32, four bases and pair metadata; "
           "per-connectivity checks remain automatic")
    select("use_packed_sm120_external_backward", packed,
           "FP16 temporal backward, SM86+, batch32 and four bases")
    # Auto keeps the existing per-connectivity fallback for oversized/int64 metadata.
    if packed and cell_params.get("use_packed_sm120_backward") is None:
        options["use_packed_sm120_backward"] = "auto"
        report["selected"]["use_packed_sm120_backward"] = "auto"
    if packed and cell_params.get("use_packed_sm120_external_backward") is None:
        options["use_packed_sm120_external_backward"] = "auto"
        report["selected"]["use_packed_sm120_external_backward"] = "auto"
    select("use_prepacked_nest_coefficients", nest and state,
           "NEST and compatible fused state")
    select(
        "use_type_indexed_nest_coefficients",
        nest and state and glif_state_ops.fused_nest_type_indexed_coefficients_available(),
        "NEST and compatible type-indexed state entry points",
    )
    carry_route = options.get("temporal_gradient_precision") == "float32" or options.get(
        "use_direct_state_rnn_loop", False
    )
    accumulator = (
        currents and small_batch and carry_route
        and architecture is not None and architecture >= 86
        and options["use_direct_csr_recurrent_gradient"] is True
        and options.get("train_recurrent", True)
        and not options.get("train_recurrent_per_type", train_recurrent_per_type)
        and pair_projection and csr_spike_ops.fused_recurrent_accumulation_available()
    )
    select("use_fused_recurrent_accumulation", accumulator,
           "SM86+ accumulator library, batch1..32, per-edge training, direct CSR "
           "and explicitly selected direct-loop or FP32 replay route")
    select(
        "use_javier_recurrent_vjp",
        accumulator and options["use_fused_recurrent_accumulation"] is True and fp16_backward,
        "compatible fused accumulator and FP16 temporal backward",
    )
    if options["use_fused_recurrent_accumulation"] is True and not accumulator:
        raise ValueError(
            "use_fused_recurrent_accumulation=True requires compatible SM86+ "
            "operators and the declared per-edge/direct-CSR carrier route."
        )
    if options["use_javier_recurrent_vjp"] is True and not (
        accumulator and options["use_fused_recurrent_accumulation"] is True and fp16_backward
    ):
        raise ValueError(
            "use_javier_recurrent_vjp=True requires compatible FP16 native accumulation."
        )
    io.log_info(f"DPointNet automatic acceleration: {report}")
    if not currents:
        io.log_warning(
            "DPointNet automatic acceleration did not admit fused CSR currents; "
            f"inspect acceleration_report for compatibility and explicit overrides: {report}"
        )
    return options, report


def resolve_weight_carry_options(
    cell_options, *, stopped_input=False, overrides=None
):
    """Select native or generic carrier accumulation for an external runner.

    Generic direct-CSR gradients retain an identity carrier and live recurrent
    credit. For stopped external inputs their unused spike VJP is zero-scaled.
    """
    requested = dict(overrides or {})
    allowed = {"native_accumulator", "compute_spike_gradient"}
    unknown = requested.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown carrier overrides: {sorted(unknown)}")
    for key, value in requested.items():
        if value is not True and value is not False:
            raise ValueError(f"{key} must be true or false.")
    architecture = csr_spike_ops._gpu_compute_architecture()
    available = (
        architecture is not None and architecture >= 86
        and csr_spike_ops.fused_recurrent_accumulation_available()
    )
    native = available and cell_options.get("use_fused_recurrent_accumulation") is True
    if cell_options.get("use_fused_recurrent_accumulation") is True and not available:
        raise ValueError("Selected native accumulator is incompatible with this GPU/library.")
    if "native_accumulator" in requested:
        native = requested["native_accumulator"]
        if native and not (
            available and cell_options.get("use_fused_recurrent_accumulation") is True
        ):
            raise ValueError("native_accumulator=True requires compatible SM86+ operators and an admitted carrier route.")
    javier = native and cell_options.get("use_javier_recurrent_vjp") is True
    weight_only = (
        native and stopped_input and javier
        and csr_spike_ops.weight_only_accumulation_available()
    )
    spike_gradient = requested.get("compute_spike_gradient", not weight_only)
    if not spike_gradient and not weight_only:
        raise ValueError(
            "compute_spike_gradient=False requires explicitly stopped inputs, "
            "rebuilt native SM86+ FP16 Javier weight-only accumulation."
        )
    if not stopped_input and not spike_gradient:
        raise ValueError("Live recurrent spike credit cannot be disabled.")
    report = {
        "requested": requested,
        "resolved": {
            "native_accumulator": native,
            "compute_spike_gradient": spike_gradient,
            "use_javier_batch32_backward": javier,
            "use_packed_sm120_backward": False,
            "use_device_active_queue_forward": cell_options.get(
                "use_device_active_queue_forward", False
            ),
            "stopped_input": stopped_input,
        },
        "reason": (
            "Native SM86+ carrier with eligible explicitly stopped input credit"
            if native else "Generic direct-CSR gradient plus TensorFlow identity carrier"
        ),
    }
    io.log_info(f"DPointNet carrier selection: {report}")
    return report


def project_weight_carry(
    spikes, carrier, csr_weights, connectivity, basis, n_post,
    spike_gradient_scale, *, resolved,
):
    """Execute a resolved carrier route without changing canonical weight order."""
    if resolved["stopped_input"]:
        spikes = tf.stop_gradient(spikes)
        spike_gradient_scale = 0.
    if resolved["native_accumulator"]:
        return csr_spike_ops.fused_recurrent_weight_carry(
            spikes, carrier, csr_weights, connectivity, basis, n_post,
            spike_gradient_scale, vjp_only=False,
            compute_spike_gradient=resolved["compute_spike_gradient"],
            use_javier_batch32_backward=resolved["use_javier_batch32_backward"],
            use_device_active_queue_forward=resolved["use_device_active_queue_forward"],
        )
    currents = csr_spike_ops.fused_spike_currents(
        spikes, carrier, csr_weights, connectivity, basis, n_post,
        compute_spike_gradient=True, spike_gradient_scale=spike_gradient_scale,
        use_packed_sm120_backward=False, write_csr_weight_gradient=True,
        use_device_active_queue_forward=resolved["use_device_active_queue_forward"],
    )
    return currents, tf.identity(carrier)
