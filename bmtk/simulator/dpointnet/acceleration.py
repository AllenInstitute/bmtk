"""Opt-in hardware-aware execution choices, independent of scientific settings."""

from numbers import Integral
import warnings

import numpy as np
import tensorflow as tf

from .custom_ops import csr_spike_ops, glif_state_ops
from .io_tools import io


def performance_projection_context(
    network, inputs, *, dt, max_delay, standard_cell, single_replica, sequence_length=None,
    bptt_training=None,
):
    """Describe the high-level runner's static projection admission boundary."""
    recurrent_delay = np.max(network["synapses"]["delays"], initial=dt)
    if max_delay is not None and max_delay > 0:
        recurrent_delay = min(recurrent_delay, max_delay)

    def history_rows(delays, n_sources):
        return (max(1, int(np.ceil(delays / dt))) + 1) * int(n_sources)

    source_rows = history_rows(recurrent_delay, network["n_nodes"])
    # External NEST delays are independent of the recurrent delay cap.
    source_rows += sum(
        history_rows(np.max(item["delays"], initial=dt), item["n_inputs"])
        for item in inputs.values()
    )
    return {
        "standard_cell": standard_cell,
        "single_replica": single_replica,
        "spike_inputs": 1 <= len(inputs) <= 7 and all(
            item.get("options", {}).get("input_type", item["input_type"])
            in ("spikes", "poisson_spikes_internal")
            and item["n_inputs"] > 0 and len(item["indices"]) > 0
            for item in inputs.values()
        ),
        "n_post": int(network["n_nodes"]),
        "source_rows_upper_bound": source_rows,
        "max_edges": max(
            [len(network["synapses"]["indices"])]
            + [len(item["indices"]) for item in inputs.values()]
        ),
        "recurrent_edges": len(network["synapses"]["indices"]),
        # Declared rollout length for static admission of prepared Poisson counts.
        "sequence_length": None if sequence_length is None else int(sequence_length),
        # None without a training engine; False for local learning rules, which use
        # canonical-order weight surfaces.
        "bptt_training": bptt_training,
    }


# Compute capabilities targeted by the mandatory fat binaries. The joint sweep is
# admitted automatically on each one whose SASS the loaded library carries.
JOINT_SWEEP_ARCHITECTURES = (61, 70, 75, 80, 86, 89, 90)
# Native small-batch routes cap at 32 samples; larger batches keep previous routes.
JOINT_SWEEP_MAX_BATCH = 32
COMBINED_TARGET_TILES = (256, 512, 1024, 2048, 3072, 4096, 5120)
# Smallest batch for which automatic adaptive routing is selected. Matched full-V1
# FR-loss arms on an RTX 3090 (20-sample medians, adaptive vs dense routing):
# B1 +5.4% slower (0.4304 vs 0.4085 s/update, a 256 wide tile did not help),
# B8 -1.6%, B16 -2.3%, B32 -2.3%. Batches 2..7 were not measured and keep the
# previous dense route. Explicit ``use_adaptive_routing=True`` is still honored.
ADAPTIVE_ROUTING_MIN_BATCH = 8


def pair_projection_predicates(
    fused_cuda, batch_size, basis_width, *, compatibility=False, defer_basis=False,
):
    """Shared current/compatibility admission, independent of dynamics and precision."""
    predicates = {
        "compatible fused CUDA currents": bool(fused_cuda),
        f"batch_size must be positive and known (found {batch_size})": (
            isinstance(batch_size, Integral) and not isinstance(batch_size, bool) and batch_size > 0
        ),
        f"the synaptic basis must have positive width (found {basis_width})": (
            defer_basis if basis_width is None else basis_width > 0
        ),
    }
    if compatibility:
        predicates["compatibility_v1 auto requires batch32 and four bases"] = (
            batch_size == 32 and basis_width == 4
        )
    return predicates


def finalize_acceleration_report(report, requested, effective, *, predicates=None, policy_predicates=None):
    """Reconcile resolver intent with cell/connectivity selections for preflight.

    ``effective`` contains booleans or named paths, never unresolved "auto".
    Rollout-dependent paths are explicitly identified, not kernel-execution proof.
    """
    if report is None:
        report = {
            "profile": requested.get("acceleration_profile"),
            "architecture": csr_spike_ops._gpu_compute_architecture(),
            "tensorflow": tf.__version__,
            "requested": dict(requested),
            "selected": {}, "reasons": {}, "provenance": {}, "automatic": {},
            "predicates": {},
        }
    report.setdefault("resolver_selected", dict(report["selected"]))
    report["selected"] = dict(effective)
    report["stage"] = "cell"
    report["predicates"].update(predicates or {})
    entries = {}
    audit = []
    for name, value in effective.items():
        supplied = name in requested
        request = requested.get(name)
        provenance = report["provenance"].get(name)
        if provenance is None:
            provenance = (
                "explicit_compatibility_override" if supplied and request == "auto"
                else "explicit_override" if supplied
                else "profile" if name in report["resolver_selected"] else "constructor_default"
            )
        checks = dict(report["predicates"].get(name, {}))
        blocked = [label for label, ok in checks.items() if not ok] if value is False else []
        policy_checks = dict((policy_predicates or {}).get(name, checks))
        automatic = report["automatic"].get(name)
        if automatic is not None:
            automatic = bool(automatic and all(policy_checks.values()))
        if value is False and supplied and request is False:
            blocked.insert(0, "explicit False override")
        entry = {
            "requested": request, "supplied": supplied, "provenance": provenance,
            "resolved": value, "predicates": checks, "blocked_by": blocked,
            "current_policy": automatic, "policy_predicates": policy_checks,
        }
        entries[name] = entry
        if value is False and automatic is True and supplied and (
            request is False or provenance == "explicit_compatibility_override"
        ):
            audit.append({
                "option": name, "classification": "avoidable_execution_override",
                "message": (
                    f"{name} is OFF due to {provenance}; current policy admits it. "
                    'Use acceleration_profile="auto" and omit the override or set it to "auto".'
                ),
            })
        elif value is False and blocked:
            audit.append({
                "option": name,
                "classification": (
                    "scientific_incompatibility"
                    if any(label in ("NEST dynamics", "compute temporal gradients") for label in blocked)
                    else "eligibility_or_dependency"
                ),
                "message": "; ".join(blocked),
            })
    report["effective"] = entries
    for name, dependencies in {
        "use_fused_recurrent_accumulation": (
            "use_pair_projection", "use_direct_csr_recurrent_gradient",
        ),
        "use_javier_recurrent_vjp": ("use_fused_recurrent_accumulation", "use_pair_projection"),
        "use_csr_master_weights": ("use_combined_current_sweep",),
        "use_uint8_synapse_types": ("use_csr_master_weights",),
        "use_adaptive_routing": ("use_uint8_synapse_types",),
    }.items():
        if name in entries:
            entries[name]["dependencies"] = {
                dependency: {
                    "resolved": entries[dependency]["resolved"],
                    "provenance": entries[dependency]["provenance"],
                }
                for dependency in dependencies if dependency in entries
            }
    report["optimization_audit"] = audit
    return report


def record_rollout_acceleration(report, selections):
    """Record the last traced rollout's path without changing execution policy."""
    if report is None:
        return
    report["rollout_selected"] = dict(selections)

# Architecture-determined limits (CUDA programming guide, technical specifications
# per compute capability). Multiprocessor counts differ between products; the
# native operator queries them and the measured kernel occupancy at launch.
DEVICE_LIMITS = {
    61: {"shared_per_block_optin": 48 * 1024, "shared_per_sm": 96 * 1024,
         "threads_per_sm": 2048, "blocks_per_sm": 32, "reserved_shared_per_block": 0},
    70: {"shared_per_block_optin": 96 * 1024, "shared_per_sm": 96 * 1024,
         "threads_per_sm": 2048, "blocks_per_sm": 32, "reserved_shared_per_block": 0},
    75: {"shared_per_block_optin": 64 * 1024, "shared_per_sm": 64 * 1024,
         "threads_per_sm": 1024, "blocks_per_sm": 16, "reserved_shared_per_block": 0},
    80: {"shared_per_block_optin": 163 * 1024, "shared_per_sm": 164 * 1024,
         "threads_per_sm": 2048, "blocks_per_sm": 32, "reserved_shared_per_block": 1024},
    86: {"shared_per_block_optin": 99 * 1024, "shared_per_sm": 100 * 1024,
         "threads_per_sm": 1536, "blocks_per_sm": 16, "reserved_shared_per_block": 1024},
    89: {"shared_per_block_optin": 99 * 1024, "shared_per_sm": 100 * 1024,
         "threads_per_sm": 1536, "blocks_per_sm": 24, "reserved_shared_per_block": 1024},
    90: {"shared_per_block_optin": 227 * 1024, "shared_per_sm": 228 * 1024,
         "threads_per_sm": 2048, "blocks_per_sm": 32, "reserved_shared_per_block": 1024},
}
_STATIC_SHARED_LIMIT = 48 * 1024
_MAX_THREADS_PER_BLOCK = 1024
# Upper bound of every combined-sweep kernel's static shared scratch beyond the
# tile accumulators (block scan, row windows, offsets, amplitudes), per block
# size and index width. Checked against the built kernels' resource usage.
_SWEEP_SCRATCH_SHARED_BYTES = {(512, 4): 8 * 1024, (512, 8): 12 * 1024,
                               (1024, 4): 16 * 1024, (1024, 8): 24 * 1024}


def combined_sweep_launch_geometry(architecture, target_tile_size, *, index_bytes=4):
    """Block shape, shared memory and validity of one combined-sweep tile on a device.

    Pure function of the compute capability, mirroring the native dispatch: tiles up
    to 2048 targets use 512 threads and static shared accumulators; wider tiles use
    dynamic shared memory (1024 threads with 32-bit routing indices) and need the
    device's per-block opt-in limit. ``resident_blocks_per_sm`` is an upper bound;
    the operator measures actual occupancy before choosing narrow/wide tiles.
    """
    tile = int(target_tile_size)
    if tile not in COMBINED_TARGET_TILES:
        raise ValueError(f"combined target tile must be one of {COMBINED_TARGET_TILES}.")
    if index_bytes not in (4, 8):
        raise ValueError("index_bytes must be 4 or 8.")
    threads = 1024 if tile > 2048 and index_bytes == 4 else 512
    accumulators = tile * 4 * 4
    scratch = _SWEEP_SCRATCH_SHARED_BYTES[(threads, index_bytes)]
    static = scratch + (accumulators if tile <= 2048 else 0)
    dynamic = accumulators if tile > 2048 else 0
    total = static + dynamic
    limits = DEVICE_LIMITS.get(architecture)
    reasons = []
    if limits is None:
        reasons.append(f"compute capability {architecture} is not a fat-binary target")
    else:
        if static > _STATIC_SHARED_LIMIT:
            reasons.append("static shared memory exceeds 48 KiB")
        if total > limits["shared_per_block_optin"]:
            reasons.append(
                f"{total} shared bytes exceed the SM{architecture} per-block limit "
                f"{limits['shared_per_block_optin']}"
            )
        if threads > _MAX_THREADS_PER_BLOCK:
            reasons.append("block exceeds 1024 threads")
    resident = 0
    if not reasons:
        per_block = total + limits["reserved_shared_per_block"]
        resident = min(
            limits["threads_per_sm"] // threads,
            limits["shared_per_sm"] // per_block,
            limits["blocks_per_sm"],
        )
    return {
        "architecture": architecture, "tile": tile, "threads": threads,
        "static_shared_bytes": static, "dynamic_shared_bytes": dynamic,
        "total_shared_bytes": total,
        "shared_per_block_optin": None if limits is None else limits["shared_per_block_optin"],
        "resident_blocks_per_sm": resident, "valid": not reasons, "reasons": reasons,
    }


def select_combined_target_tile(
    architecture, batch_size, *, adaptive, index_bytes=4, minimum_tile=None,
):
    """Automatic wide tile: 2048 targets, or 256 for batch 1 without adaptive routing.

    Adaptive routing derives its per-launch narrow tile (a quarter, at least 256) from
    the wide tile, so it keeps 2048 at batch 1 where the narrow 512 shape is selected
    on the device. Falls back to the widest valid narrower tile on smaller devices.
    ``minimum_tile`` is the narrowest tile whose routing table fits the 512 MiB
    budget of ``select_performance_projection``; narrower tiles are never chosen
    (None when no tile at least that wide is launchable).
    """
    floor = COMBINED_TARGET_TILES[0] if minimum_tile is None else int(minimum_tile)
    preferred = max(2048 if adaptive or batch_size != 1 else 256, floor)
    for tile in sorted(
        (t for t in COMBINED_TARGET_TILES if floor <= t <= preferred), reverse=True,
    ):
        if combined_sweep_launch_geometry(architecture, tile, index_bytes=index_bytes)["valid"]:
            return tile
    return None


def _index_bytes(context):
    return 4 if max(
        context["n_post"], context["source_rows_upper_bound"], context["max_edges"]
    ) <= 2**32 - 1 else 8


def select_performance_projection(
    options, context, *, compute_dtype, variable_dtype, batch_size, basis_width,
    architecture, diagnostics=None,
):
    """Choose a qualified joint-current route without changing explicit options."""
    reason = (
        "single-replica standard GLIF/NEST, fat-binary SM61/70/75/80/86/89/90 with native "
        "SASS, FP16/FP32 with FP32 masters, batch1..32, four bases, native spike inputs and "
        "bounded routing; joint FP32 accumulation rounds once to compute dtype"
    )
    context = context or {}
    predicates = {
        "automatic profile": options.get("acceleration_profile", "auto") == "auto",
        "joint sweep not explicitly disabled": options.get("use_combined_current_sweep") is not False,
        "fused currents not explicitly disabled": options.get("use_fused_cuda") is not False,
        "NEST dynamics": options.get("dynamics_mode", "nest") == "nest",
        "compute temporal gradients": options.get("temporal_gradient_precision", "compute") == "compute",
        f"SM{architecture} in qualified fat-binary targets": architecture in JOINT_SWEEP_ARCHITECTURES,
        f"SM{architecture} SASS in loaded library": architecture in csr_spike_ops._SM_ARCHITECTURES,
        "FP16/FP32 compute": tf.as_dtype(compute_dtype) in (tf.float16, tf.float32),
        "FP32 masters": tf.as_dtype(variable_dtype) == tf.float32,
        f"batch1..32 (found {batch_size})": (
            isinstance(batch_size, Integral) and not isinstance(batch_size, bool)
            and 1 <= batch_size <= JOINT_SWEEP_MAX_BATCH
        ),
        f"four bases (found {basis_width})": basis_width == 4,
        "standard high-level cell": context.get("standard_cell", False),
        "single replica": context.get("single_replica", False),
        "nonempty supported spike inputs": context.get("spike_inputs", False),
        "nonempty recurrence": context.get("recurrent_edges", 0) > 0,
        "loaded joint-sweep ABI": csr_spike_ops.combined_sweep_available(),
    }
    eligible = all(predicates.values())
    tile = 256 if eligible and batch_size == 1 else 2048
    if eligible:
        index_bytes = _index_bytes(context)

        def routing_bytes(targets):
            return (
                ((context["n_post"] + targets - 1) // targets + 1)
                * context["source_rows_upper_bound"] * index_bytes
            )

        if routing_bytes(tile) > 512 * 1024**2:
            tile = 2048
        eligible = routing_bytes(tile) <= 512 * 1024**2 and combined_sweep_launch_geometry(
            architecture, tile, index_bytes=index_bytes,
        )["valid"]
        predicates["routing table <= 512 MiB and launchable tile"] = eligible
    missing = [name for name, ok in predicates.items() if not ok]
    if missing:
        reason += "; blocked by: " + "; ".join(missing)
    if diagnostics is not None:
        diagnostics.update(predicates)
    return bool(eligible), tile, reason


def _option_names_requiring(missing):
    return "; ".join(missing)


def resolve_csr_master_options(
    options, report, *, select, admitted, joint, variable_dtype, train_recurrent_per_type,
    performance_context, synapse_type_range, batch_size=None,
):
    """Resolve the CSR-master layout, uint8 type tail and adaptive routing options.

    Omitted options enable each feature exactly where its prerequisites hold and
    otherwise keep the previous path, recording the unmet prerequisites in
    ``report["reasons"]``. Explicit ``True`` on a dependent option whose parent
    resolved to ``False`` raises; explicit ``False`` restores the previous path.
    """
    missing = []
    if not admitted or not joint:
        missing.append("automatically admitted joint current sweep")
    if performance_context is None or not performance_context.get("single_replica", False):
        missing.append("single-replica standard high-level context")
    elif tf.distribute.get_strategy().num_replicas_in_sync != 1:
        missing.append("single replica")
    if options.get("dynamics_mode", "nest") != "nest":
        missing.append("NEST dynamics")
    if options.get("temporal_gradient_precision", "compute") != "compute":
        missing.append("compute temporal gradients")
    if tf.as_dtype(variable_dtype) != tf.float32:
        missing.append("FP32 master variables")
    if not options.get("train_recurrent", True) or options.get(
        "train_recurrent_per_type", train_recurrent_per_type
    ):
        missing.append("per-edge trainable recurrent weights")
    if performance_context is not None and performance_context.get("bptt_training") is False:
        missing.append("BPTT learning rule (local rules use canonical surfaces)")
    if not csr_spike_ops.csr_master_weights_available():
        missing.append("native CSR-master capability in the loaded library")
    select(
        "use_csr_master_weights", not missing,
        "CSR-ordered FP32 masters read directly by native kernels with host-prepared "
        "metadata" if not missing else "previous path kept; requires " + _option_names_requiring(missing),
        {name: False for name in missing} if missing else {"CSR-master prerequisites": True},
    )
    masters = options["use_csr_master_weights"] is True

    missing = [] if masters else ["use_csr_master_weights"]
    if not admitted:
        missing.append("automatically admitted joint current sweep")
    if synapse_type_range is None:
        missing.append("known synapse type IDs")
    elif not (0 <= synapse_type_range[0] and synapse_type_range[1] <= 255):
        missing.append(
            f"synapse type IDs within 0..255 (found {synapse_type_range[0]}..{synapse_type_range[1]})"
        )
    if not csr_spike_ops.uint8_synapse_types_available():
        missing.append("native uint8 type capability on all five type consumers")
    if options.get("use_uint8_synapse_types") is True and not masters:
        raise ValueError(
            "use_uint8_synapse_types=True requires use_csr_master_weights, which resolved to "
            "False: " + report["reasons"]["use_csr_master_weights"]
        )
    select(
        "use_uint8_synapse_types", not missing,
        "uint8 per-edge type tail in the single CSR metadata resource" if not missing
        else "wide types kept; requires " + _option_names_requiring(missing),
        {name: False for name in missing} if missing else {"uint8 prerequisites": True},
    )
    uint8 = options["use_uint8_synapse_types"] is True

    missing = [] if uint8 else ["use_uint8_synapse_types"]
    if not admitted or not joint:
        missing.append("automatically admitted joint current sweep")
    if not csr_spike_ops.adaptive_routing_available():
        missing.append("native adaptive routing capability")
    if options.get("use_adaptive_routing") is True and not uint8:
        raise ValueError(
            "use_adaptive_routing=True requires use_uint8_synapse_types, which resolved to "
            "False: " + report["reasons"]["use_uint8_synapse_types"]
        )
    if (
        not missing and isinstance(batch_size, Integral) and not isinstance(batch_size, bool)
        and batch_size < ADAPTIVE_ROUTING_MIN_BATCH
    ):
        reason = (
            f"dense tile routing kept; automatic adaptive routing requires batch >= "
            f"{ADAPTIVE_ROUTING_MIN_BATCH} (measured slower at batch 1, unmeasured below "
            f"{ADAPTIVE_ROUTING_MIN_BATCH}); explicit True is honored"
        )
        enabled = False
    else:
        enabled = not missing
        reason = (
            "compressed segment routing with per-launch wide/narrow tiles" if not missing
            else "dense tile routing kept; requires " + _option_names_requiring(missing)
        )
    checks = {name: False for name in missing}
    checks[f"batch >= {ADAPTIVE_ROUTING_MIN_BATCH} (found {batch_size})"] = (
        isinstance(batch_size, Integral) and batch_size >= ADAPTIVE_ROUTING_MIN_BATCH
    )
    select("use_adaptive_routing", enabled, reason, checks)


def resolve_acceleration_options(
    cell_params, *, compute_dtype, variable_dtype, batch_size, basis_width,
    train_recurrent_per_type=False, canonical_gradient_boundary=True,
    performance_context=None, synapse_type_range=None,
):
    """Resolve automatic requests centrally, retaining explicit boolean overrides.

    The returned report is suitable for recording with execution provenance.
    Omission and per-option "auto" use the same current policy under auto.
    None and "compatibility_v1" disable expansion (historical constructor policy).
    ``synapse_type_range`` is the (min, max) per-edge synapse type ID over the
    recurrent and every input network, used by ``use_uint8_synapse_types``.
    """
    options = dict(cell_params)
    profile = options.pop("acceleration_profile", "auto")
    if profile is None or profile == "compatibility_v1":
        if profile is None and any(
            name.startswith("use_") and value == "auto" for name, value in options.items()
        ):
            warnings.warn(
                'acceleration_profile=None retains historical per-option "auto" heuristics. '
                'Use "compatibility_v1" to name this policy, or "auto" for current policy.',
                FutureWarning, stacklevel=2,
            )
        return options, None
    if profile != "auto":
        raise ValueError('acceleration_profile must be None, "auto" or "compatibility_v1".')

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
        "provenance": {},
        "automatic": {},
        "predicates": {},
        "stage": "resolver",
    }

    def select(name, enabled, reason, predicates=None):
        report["automatic"][name] = bool(enabled)
        report["predicates"][name] = predicates or {reason: bool(enabled)}
        if name in options and options[name] != "auto":
            report["reasons"][name] = "explicit override; constructor validation retained"
            report["provenance"][name] = "explicit_override"
        else:
            options[name] = bool(enabled)
            report["reasons"][name] = reason
            report["provenance"][name] = (
                "explicit_auto" if name in cell_params else "profile"
            )
        report["selected"][name] = options[name]

    numeric = compute_dtype in (tf.float16, tf.float32) and variable_dtype == tf.float32
    nest = options.get("dynamics_mode", "nest") == "nest"
    pascal = architecture == 61
    unsupported_nest = nest and architecture is not None and architecture < 70 and not pascal
    currents = numeric and csr_spike_ops.fused_cuda_available() and not unsupported_nest
    select(
        "use_fused_cuda", currents,
        "compatible loaded CSR library and FP16/FP32 compute with FP32 masters; "
        "SM61-compatible CSR currents; matching loaded library, FP16/FP32 compute "
        "and FP32 masters; explicit homogeneous data-parallel GPU strategies supported",
        {
            "FP16/FP32 compute and FP32 masters": numeric,
            "loaded compatible CSR library": csr_spike_ops.fused_cuda_available(),
            "supported NEST architecture": not unsupported_nest,
        },
    )
    currents = currents and options["use_fused_cuda"] is not False
    projection_checks = {}
    projection_options = dict(cell_params)
    projection_options.pop("use_combined_current_sweep", None)
    combined, tile, combined_reason = select_performance_projection(
        projection_options, performance_context, compute_dtype=compute_dtype,
        variable_dtype=variable_dtype, batch_size=batch_size, basis_width=basis_width,
        architecture=architecture, diagnostics=projection_checks,
    )
    projection_checks["compatible fused currents"] = currents
    select("use_combined_current_sweep", currents and combined, combined_reason, projection_checks)
    combined = combined and cell_params.get("use_combined_current_sweep") is not False
    joint = options["use_combined_current_sweep"] is True
    # Projection traversal choices inside the admitted joint sweep. Each is a
    # pure scheduling change over the same FP32 tile accumulation and rounding;
    # per-source VJPs, live operands, delays and Poisson counts are unchanged.
    # Explicit False keeps the original traversal (compatibility override).
    select(
        "use_cooperative_row_projection",
        joint and csr_spike_ops.cooperative_combined_sweep_available(),
        "admitted joint sweep and rebuilt cooperative-row capability; register-only "
        "eight-row warp packing for known target-unique sources below512edges per row",
        {"joint sweep": joint, "loaded cooperative-row ABI": csr_spike_ops.cooperative_combined_sweep_available()},
    )
    select(
        "use_fanout_row_projection",
        joint and csr_spike_ops.fanout_combined_sweep_available(),
        "admitted joint sweep and rebuilt fanout-row capability; compact nonempty-row "
        "scan with concatenated edge walk for sources of at least512edges per row",
        {"joint sweep": joint, "loaded fanout-row ABI": csr_spike_ops.fanout_combined_sweep_available()},
    )
    select(
        "use_target_gather_projection",
        joint and csr_spike_ops.gather_combined_sweep_available(),
        "admitted joint sweep and rebuilt gather capability; target-major atomic-free "
        "gather for sources with at most16incoming edges per target",
        {"joint sweep": joint, "loaded target-gather ABI": csr_spike_ops.gather_combined_sweep_available()},
    )
    resolve_csr_master_options(
        options, report, select=select, admitted=currents and combined, joint=joint,
        variable_dtype=variable_dtype, train_recurrent_per_type=train_recurrent_per_type,
        performance_context=performance_context, synapse_type_range=synapse_type_range,
        batch_size=batch_size,
    )
    if joint:
        tile_name = "combined_current_target_tile_size"
        if tile_name in options:
            report["reasons"][tile_name] = "explicit override; constructor validation retained"
            if architecture in DEVICE_LIMITS and options[tile_name] in COMBINED_TARGET_TILES:
                geometry = combined_sweep_launch_geometry(
                    architecture, options[tile_name],
                    index_bytes=4 if performance_context is None else _index_bytes(performance_context),
                )
                if not geometry["valid"]:
                    valid = [
                        t for t in COMBINED_TARGET_TILES
                        if combined_sweep_launch_geometry(architecture, t)["valid"]
                    ]
                    raise ValueError(
                        f"combined_current_target_tile_size={options[tile_name]} is not launchable "
                        f"on SM{architecture}: {'; '.join(geometry['reasons'])}. "
                        f"Valid tiles on this device: {valid}."
                    )
        else:
            adaptive = options.get("use_adaptive_routing") is True
            auto_tile = tile
            if combined:
                # The projection's tile is the narrowest one whose routing table fits
                # the budget; without adaptive routing it is the previous route's tile.
                auto_tile = select_combined_target_tile(
                    architecture, batch_size, adaptive=adaptive,
                    index_bytes=_index_bytes(performance_context), minimum_tile=tile,
                ) or tile
            options[tile_name] = auto_tile
            report["reasons"][tile_name] = (
                "batch-aware bounded target routing; 2048 wide tiles (adaptive routing picks "
                "a narrow quarter tile per launch) or 256 at batch1 without adaptive routing "
                "when its routing table fits 512 MiB (else 2048); narrower when the device's "
                "shared-memory limit requires it, never below the routing budget"
            )
        report["selected"][tile_name] = options[tile_name]
        report["launch_geometry"] = (
            combined_sweep_launch_geometry(
                architecture, options[tile_name],
                index_bytes=_index_bytes(performance_context),
            )
            if architecture in DEVICE_LIMITS and performance_context is not None
            and options[tile_name] in COMBINED_TARGET_TILES else None
        )
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
    if nest and pascal:
        state_available = state_available and glif_state_ops.fused_pascal_nest_launch_available()
    select(
        "use_fused_state",
        numeric and four_basis and state_available and not unsupported_nest,
        "compatible mode-specific state library, FP32 masters and four bases; "
        "SM61 NEST requires rebuilt occupancy-aware launch capability",
        {
            "FP16/FP32 compute and FP32 masters": numeric,
            "four synaptic bases": four_basis,
            "mode-specific state ABI (including Pascal occupancy)": state_available,
            "supported NEST architecture": not unsupported_nest,
        },
    )
    state = options["use_fused_state"] is not False and numeric and four_basis and state_available and not unsupported_nest
    select("use_direct_state_rnn_loop", state and currents,
           "compatible fused state/currents and explicit-state loop",
           {"compatible fused state": state, "compatible fused currents": currents})
    prepared = (
        nest and options["use_direct_state_rnn_loop"] is True
        and options.get("temporal_gradient_precision", "compute") == "compute"
        and options.get("use_device_poisson") is not True
        and performance_context is not None and performance_context["single_replica"]
        and performance_context["standard_cell"]
    )
    if "use_prepared_cpu_poisson" in options and options["use_prepared_cpu_poisson"] != "auto":
        report["reasons"]["use_prepared_cpu_poisson"] = "explicit override; constructor validation retained"
    else:
        # "auto" prepares exact per-step CPU Poisson counts once per rollout when
        # the RNN layer actually runs the direct NEST loop within the static
        # count budget, and otherwise keeps the ordinary per-step sampler.
        options["use_prepared_cpu_poisson"] = "auto" if prepared else False
        report["reasons"]["use_prepared_cpu_poisson"] = (
            "NEST direct loop, compute temporal gradients, internal Poisson, single replica; "
            "the layer falls back to per-step sampling outside the direct loop or budget"
        )
    report["selected"]["use_prepared_cpu_poisson"] = options["use_prepared_cpu_poisson"]
    report["automatic"]["use_prepared_cpu_poisson"] = prepared
    report["provenance"]["use_prepared_cpu_poisson"] = (
        "explicit_auto" if cell_params.get("use_prepared_cpu_poisson") == "auto"
        else "explicit_override" if "use_prepared_cpu_poisson" in cell_params else "profile"
    )
    select("use_fused_nest_event_vjp",
           nest and state and glif_state_ops.fused_nest_event_vjp_available(),
           "compatible NEST state and loaded event-gradient operator")
    select("use_fused_state_history",
           nest and state and glif_state_ops.fused_nest_state_history_available(),
           "compatible NEST state and loaded history operators")
    select("use_native_voltage_penalty",
           state and options.get("track_voltage_penalty", True)
           and glif_state_ops.fused_voltage_penalty_available(),
           "compatible state, enabled penalty tracking and loaded penalty operators")
    select("use_forward_run_aggregation", currents and small_batch and four_basis,
           "compatible batch1..32/four-basis CSR route")
    pair_predicates = pair_projection_predicates(
        currents, batch_size, basis_width, defer_basis=True,
    )
    select("use_pair_projection", all(pair_predicates.values()),
           "compatible CSR library, known positive batch and nonempty basis", pair_predicates)
    select("use_fixed4_input_forward", currents and four_basis,
           "compatible CSR library and four bases; each input checks exactly four incoming edges")
    select("use_fused_current_accumulation", currents and four_basis and batch_size == 32,
           "compatible CSR library, four bases and batch32")
    per_edge_training = options.get("train_recurrent", True) and not options.get(
        "train_recurrent_per_type", train_recurrent_per_type
    )
    select("use_direct_csr_recurrent_gradient", currents and per_edge_training and canonical_gradient_boundary,
           "compatible CSR library, per-edge weights and an enclosing canonical-gradient boundary",
           {
               "compatible fused currents": currents, "per-edge trainable recurrence": per_edge_training,
               "canonical-gradient boundary": canonical_gradient_boundary,
           })
    select("use_active_row_forward", currents and small_batch and four_basis,
           "compatible CSR library, batch1..32 and four bases")
    select("use_device_active_queue_forward", currents and small_batch and four_basis,
           "compatible CSR library, batch1..32 and four bases")
    pair_projection = options["use_pair_projection"] is True
    packed_predicates = {
        "compatible fused currents": currents,
        "FP16 compute-temporal backward": fp16_backward,
        "four synaptic bases": four_basis,
        f"batch32 (found {batch_size})": batch_size == 32,
        f"SM{architecture} admitted for automatic packed backward": (
            csr_spike_ops._auto_native_architecture(architecture)
        ),
        "use_pair_projection=True": pair_projection,
    }
    packed = all(packed_predicates.values())
    select("use_packed_sm120_backward", packed,
           "FP16 temporal backward, automatic policy SM70+, batch32, four bases and pair metadata; "
           "per-connectivity checks remain automatic", packed_predicates)
    select("use_packed_sm120_external_backward", packed,
           "FP16 temporal backward, automatic policy SM70+, batch32 and four bases", packed_predicates)
    # Auto keeps the existing per-connectivity fallback for oversized/int64 metadata.
    if packed and cell_params.get("use_packed_sm120_backward") in (None, "auto"):
        options["use_packed_sm120_backward"] = "auto"
        report["selected"]["use_packed_sm120_backward"] = "auto"
    if packed and cell_params.get("use_packed_sm120_external_backward") in (None, "auto"):
        options["use_packed_sm120_external_backward"] = "auto"
        report["selected"]["use_packed_sm120_external_backward"] = "auto"
    select("use_prepacked_nest_coefficients", nest and state,
           "NEST and compatible fused state")
    select(
        "use_type_indexed_nest_coefficients",
        nest and state and (
            glif_state_ops.fused_nest_state_history_type_indexed_available()
            if options["use_fused_state_history"] is True
            else glif_state_ops.fused_nest_type_indexed_coefficients_available()
        ),
        "NEST and compatible type-indexed entry points for the selected state/history route",
    )
    select("use_static_type_indexed_nest_dispatch", False,
           "static coefficient identity requires explicit opt-in; auto rechecks live coefficients")
    carry_route = options.get("temporal_gradient_precision") == "float32" or options.get(
        "use_direct_state_rnn_loop", False
    )
    accumulator_predicates = {
        "compatible fused currents": currents,
        f"batch1..32 (found {batch_size})": small_batch,
        f"four bases (found {basis_width})": four_basis,
        "direct state loop or FP32 temporal carry": bool(carry_route),
        f"SM{architecture} admitted for accumulator": (
            pascal or csr_spike_ops._auto_native_architecture(architecture)
        ),
        "SM75 requires FP16 temporal backward": architecture != 75 or fp16_backward,
        "use_direct_csr_recurrent_gradient=True": options["use_direct_csr_recurrent_gradient"] is True,
        "per-edge trainable recurrence": bool(per_edge_training),
        "use_pair_projection=True": pair_projection,
        "loaded fused accumulator ABI": csr_spike_ops.fused_recurrent_accumulation_available(),
    }
    accumulator = all(accumulator_predicates.values())
    select("use_fused_recurrent_accumulation", accumulator,
           "Automatic policy SM61, SM70+ (SM75 with FP16 temporal backward) accumulator library, "
           "batch1..32/four bases, per-edge training, direct CSR "
           "and explicitly selected direct-loop or FP32 replay route", accumulator_predicates)
    select(
        "use_javier_recurrent_vjp",
        accumulator and options["use_fused_recurrent_accumulation"] is True and fp16_backward,
        "compatible fused accumulator and FP16 temporal backward",
        {
            "accumulator eligible": accumulator,
            "use_fused_recurrent_accumulation=True": options["use_fused_recurrent_accumulation"] is True,
            "FP16 compute-temporal backward": fp16_backward,
        },
    )
    if options["use_fused_recurrent_accumulation"] is True and not accumulator:
        raise ValueError(
            "use_fused_recurrent_accumulation=True with auto requires an admitted SM61/SM70+ GPU "
            "operators and the declared per-edge/direct-CSR carrier route."
        )
    if options["use_javier_recurrent_vjp"] is True and not (
        accumulator and options["use_fused_recurrent_accumulation"] is True and fp16_backward
    ):
        raise ValueError(
            "use_javier_recurrent_vjp=True requires compatible FP16 native accumulation."
        )
    report["resolver_selected"] = dict(report["selected"])
    io.log_info(f"DPointNet acceleration resolver (not final cell dispatch): {report}")
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
        architecture is not None and architecture >= 61
        and csr_spike_ops.fused_cuda_available()
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
            raise ValueError("native_accumulator=True requires compatible SM61+ operators and an admitted carrier route.")
    javier = native and cell_options.get("use_javier_recurrent_vjp") is True
    weight_only = (
        native and stopped_input and javier
        and csr_spike_ops.weight_only_accumulation_available()
    )
    spike_gradient = requested.get("compute_spike_gradient", not weight_only)
    if not spike_gradient and not weight_only:
        raise ValueError(
            "compute_spike_gradient=False requires explicitly stopped inputs, "
            "rebuilt native SM61+ FP16 Javier weight-only accumulation."
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
            "Native SM61+ carrier with eligible explicitly stopped input credit"
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
