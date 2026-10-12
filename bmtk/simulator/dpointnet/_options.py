"""Shared option parsing without changing individual option contracts."""

import warnings

import numpy as np


UNSET = object()


def resolve_renamed_option(
    name, value, legacy_name, legacy_value, default, transform=None, *, stacklevel=2
):
    if value is not UNSET and legacy_value is not UNSET:
        raise ValueError(f"Specify only {name} or its legacy alias {legacy_name}, not both.")
    if legacy_value is not UNSET:
        resolved = transform(legacy_value) if transform else legacy_value
        warnings.warn(
            f"{legacy_name} is deprecated; use {name} instead. "
            "The legacy alias retains its existing value-conversion semantics; "
            "see the gradient-control migration table before renaming it. "
            "No removal version is scheduled.",
            FutureWarning,
            stacklevel=stacklevel,
        )
        return resolved
    return default if value is UNSET else value


def resolve_precision_options(dtype, precision, cell_params):
    options = dict(cell_params)
    if precision is not None and not isinstance(precision, dict):
        raise ValueError("run.precision must be a dictionary of boolean switches.")
    switches = {} if precision is None else dict(precision)
    allowed = {"mixed_precision", "fp32_voltage_and_asc", "fp32_temporal_gradients"}
    unknown = switches.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown precision switches: {sorted(unknown)}")
    for name, value in switches.items():
        validate_bool_option(value, f"precision.{name}")
    resolved_dtype = "float16" if dtype is UNSET else dtype
    if "mixed_precision" in switches:
        selected = "float16" if switches["mixed_precision"] else "float32"
        if dtype is not UNSET and dtype != selected:
            raise ValueError("run.dtype conflicts with precision.mixed_precision.")
        resolved_dtype = selected
    mixed = resolved_dtype == "float16"
    state = options.get("state_precision", "selective" if mixed else "compute")
    temporal = options.get("temporal_gradient_precision", "compute")
    if "fp32_voltage_and_asc" in switches:
        selected = "selective" if mixed and switches["fp32_voltage_and_asc"] else "compute"
        if "state_precision" in options and state != selected:
            raise ValueError("state_precision conflicts with precision.fp32_voltage_and_asc.")
        state = selected
    if "fp32_temporal_gradients" in switches:
        selected = "float32" if mixed and switches["fp32_temporal_gradients"] else "compute"
        if "temporal_gradient_precision" in options and temporal != selected:
            raise ValueError(
                "temporal_gradient_precision conflicts with precision.fp32_temporal_gradients."
            )
        temporal = selected
    if mixed and temporal == "float32" and state != "selective":
        raise ValueError("FP32 temporal gradients require fp32_voltage_and_asc=True.")
    options["state_precision"] = state
    options["temporal_gradient_precision"] = temporal
    report = {
        "compute_dtype": resolved_dtype,
        "state_precision": state,
        "temporal_gradient_precision": temporal,
        "requested_switches": switches,
    }
    return resolved_dtype, options, report


def validate_bool_option(value, name, *, allow_auto=False, unwrap_numpy=False):
    if unwrap_numpy and isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is True or value is False:
        return value
    if allow_auto:
        if isinstance(value, (bytes, np.bytes_)):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                pass
        if isinstance(value, (str, np.str_)) and value == "auto":
            return "auto"
        raise ValueError(f'{name} must be true, false, or "auto".')
    raise ValueError(f"{name} must be true or false.")
