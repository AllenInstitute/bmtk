"""Explicit preservation of pre-standard scientific settings in example configs."""

from copy import deepcopy
from math import isfinite


def preserve_previous_defaults(config):
    result = deepcopy(config)
    cell = result.get("rnn_cell_params")
    if not isinstance(cell, dict) or cell.get("cell_model", "GLIF3Cell") not in ("GLIF3Cell", "GLIF3", "default"):
        return result
    if result.get("training"):
        result["training"].setdefault("gradient_checkpointing", False)
    run = result.setdefault("run", {})
    if "dtype" not in run and "precision" not in run:
        run["dtype"] = "float32"
    if run.get("dtype") == "float16" and "precision" not in run:
        cell.setdefault("state_precision", "compute")
    aliases = {
        "gauss_std": "spike_surrogate_width",
        "dampening_factor": "spike_surrogate_gain",
        "recurrent_dampening_factor": "recurrent_spike_gradient_scale",
        "voltage_gradient_dampening": "voltage_state_gradient_scale",
        "pseudo_gauss": "spike_surrogate",
    }
    for old, new in aliases.items():
        if old not in cell:
            continue
        if new in cell:
            raise ValueError(f"Specify only {new} or its legacy alias {old}, not both.")
        value = cell.pop(old)
        if old == "voltage_gradient_dampening":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError("voltage_gradient_dampening must be numeric.")
            value = 1.0 - min(max(value, 0.0), 1.0)
        elif old == "pseudo_gauss":
            if value is not True and value is not False:
                raise ValueError("pseudo_gauss must be true or false.")
            value = "gaussian" if value else "triangular"
        cell[new] = value
    cell.setdefault("dynamics_mode", "legacy")
    cell.setdefault("spike_surrogate", "triangular")
    if cell["spike_surrogate"] == "gaussian":
        cell.setdefault("spike_surrogate_width", 0.5)
    cell.setdefault("spike_surrogate_gain", 0.3)
    cell.setdefault("recurrent_spike_gradient_scale", 0.5)
    cell.setdefault("voltage_state_gradient_scale", 0.5)
    cell.setdefault("detach_asc_reset", True)
    cell.setdefault("track_voltage_penalty", False)
    cell.setdefault("return_voltage_sequences", True)
    if (
        cell["dynamics_mode"] == "nest"
        and not result.get("training")
        and cell.get("hard_reset") is None
    ):
        cell["hard_reset"] = True
    cell.setdefault("hard_reset", False)
    cell.setdefault("hard_reset_gradient_mode", "exact")
    cell["acceleration_profile"] = "auto"
    return result
