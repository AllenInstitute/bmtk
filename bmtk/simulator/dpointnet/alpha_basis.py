"""Fit shared alpha time constants and per-waveform linear coefficients."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

DEFAULT_ALPHA_BASIS_TOLERANCE = 0.08012288897995931


def alpha_design(time, tau_basis):
    """Return peak-normalized alpha functions as columns."""
    scaled_time = np.asarray(time)[:, None] / np.asarray(tau_basis)[None, :]
    return scaled_time * np.exp(1.0 - scaled_time)


def fit_alpha_basis(
    parameters,
    tolerance=DEFAULT_ALPHA_BASIS_TOLERANCE,
    min_basis=4,
    max_basis=5,
    time_range=None,
    n_points=1000,
    seed=42,
    max_iterations=100,
    tau_basis=None,
):
    """Fit (fast tau, slow tau, slow amplitude) rows by variable projection.

    Linear coefficients are unconstrained, as in Javier Galvan's alpha-basis
    notebook. Shared time constants minimize mean waveform MSE. The smallest
    basis meeting the worst-class relative RMS tolerance is returned. Failure
    at the configured maximum raises rather than silently relaxing tolerance.
    """
    started = perf_counter()
    parameters = np.asarray(parameters, dtype=np.float64)
    if (
        parameters.ndim != 2
        or parameters.shape[1] != 3
        or len(parameters) == 0
        or not np.isfinite(parameters).all()
        or np.any(parameters[:, :2] <= 0)
    ):
        raise ValueError(
            "parameters must contain finite positive fast/slow taus and amplitudes"
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    integer_options = (min_basis, max_basis, n_points, max_iterations, seed)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in integer_options
    ):
        raise ValueError(
            "Basis counts, sample count, optimization budget and seed must be integers"
        )
    if (
        not 1 <= min_basis <= max_basis
        or n_points < 10
        or max_iterations < 1
        or seed < 0
    ):
        raise ValueError("Invalid basis count, sample count, or optimization budget")
    if time_range is None:
        time_range = max(30.0, 10.0 * parameters[:, :2].max())
    if not np.isfinite(time_range) or time_range <= 0:
        raise ValueError("time_range must be finite and positive")
    time = np.linspace(0.0, time_range, n_points)
    targets = np.array(
        [alpha_design(time, row[:2]) @ np.array([1.0, row[2]]) for row in parameters]
    )
    energy = np.mean(targets**2, axis=1)
    if np.any(energy <= 0):
        raise ValueError("Cannot fit a zero-energy synaptic waveform")

    def evaluate(taus):
        design = alpha_design(time, taus)
        weights = np.linalg.lstsq(design, targets.T, rcond=None)[0].T
        mse = np.mean((targets - weights @ design.T) ** 2, axis=1)
        return weights, mse

    attempts = []
    if tau_basis is not None:
        tau_basis = np.asarray(tau_basis, dtype=np.float64)
        if (
            tau_basis.ndim != 1
            or not len(tau_basis)
            or not np.isfinite(tau_basis).all()
            or np.any(tau_basis <= 0)
        ):
            raise ValueError("tau_basis must be a finite positive vector")
        counts = [len(tau_basis)]
    else:
        counts = range(min_basis, max_basis + 1)
    log_bounds = [
        (np.log(parameters[:, :2].min() / 10.0), np.log(parameters[:, :2].max() * 10.0))
    ]
    for count in counts:
        if tau_basis is None:
            bounds = log_bounds * count

            def objective(log_taus):
                return evaluate(np.exp(log_taus))[1].mean()

            result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=max_iterations,
                popsize=8,
                tol=1e-7,
                atol=1e-12,
                polish=False,
                workers=1,
            )
            initial = np.linspace(
                np.log(parameters[:, :2].min()), np.log(parameters[:, :2].max()), count
            )
            candidates = [result.x, initial]
            if attempts:
                candidates.append(
                    np.log(
                        np.append(
                            attempts[-1]["tau_basis"],
                            np.sqrt(parameters[:, :2].min() * parameters[:, :2].max()),
                        )
                    )
                )
            best = result
            for candidate in candidates:
                refined = minimize(
                    objective,
                    candidate,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_iterations * 5, "ftol": 1e-12},
                )
                if refined.fun < best.fun:
                    best = refined
            taus = np.sort(np.exp(best.x))
        else:
            taus = tau_basis
        weights, mse = evaluate(taus)
        relative_rms = np.sqrt(mse / energy)
        attempts.append(
            {
                "n_basis": count,
                "tau_basis": taus.tolist(),
                "mean_mse": float(mse.mean()),
                "max_relative_rms": float(relative_rms.max()),
            }
        )
        if relative_rms.max() <= tolerance:
            return {
                "tau_basis": taus,
                "weights": weights,
                "mse": mse,
                "relative_rms": relative_rms,
                "diagnostics": {
                    "method": "shared-tau-variable-projection",
                    "tolerance": float(tolerance),
                    "time_range_ms": float(time_range),
                    "n_points": n_points,
                    "seed": seed,
                    "attempts": attempts,
                    "elapsed_seconds": perf_counter() - started,
                },
            }
    raise ValueError(
        "Alpha basis did not meet relative RMS tolerance {} with {} functions (error {}). "
        "Inspect kinetics or explicitly configure a larger basis/budget.".format(
            tolerance,
            counts[-1] if isinstance(counts, list) else max_basis,
            attempts[-1]["max_relative_rms"],
        )
    )


def _target_types_by_edge(population, target):
    node_population = target._sonata_node_pop
    node_ids = pd.Index(node_population.node_ids)
    node_types = np.asarray(node_population.type_ids)
    type_ids, type_indices = np.unique(node_types, return_inverse=True)
    target_types = {}
    count = len(population._type_id_ds)
    for start in range(0, count, 262144):
        target_ids = population._target_node_id_ds[start : start + 262144]
        positions = node_ids.get_indexer(target_ids)
        if np.any(positions < 0):
            raise ValueError("Unmapped SONATA target node IDs")
        edge_types = np.asarray(
            population._type_id_ds[start : start + 262144], dtype=np.int64
        )
        keys = edge_types * len(type_ids) + type_indices[positions]
        for key in np.unique(keys):
            edge_type, type_index = divmod(int(key), len(type_ids))
            target_types.setdefault(edge_type, set()).add(type_ids[type_index])
    return target_types


def prepare_alpha_basis(networks, cell_params, options=None):
    """Populate missing SONATA coefficients without modifying input files."""
    if options is False:
        return None
    options = {} if options is None or options is True else dict(options)
    force_recompute = options.pop("force_recompute", False)
    if not isinstance(force_recompute, bool):
        raise ValueError("alpha_basis.force_recompute must be true or false")
    if not force_recompute and cell_params.get("synaptic_basis_weights") is not None:
        return None
    tau_basis = None if force_recompute else cell_params.get("tau_basis")
    if isinstance(tau_basis, (str, Path)):
        tau_basis = np.load(tau_basis)
    records = {}
    for network in networks:
        for population in getattr(network, "_sonata_edge_pops", []):
            for type_id in np.unique(population.type_ids):
                descriptor = population.types_table[type_id]
                filename = Path(descriptor["dynamics_params"])
                if filename not in records:
                    with filename.open() as handle:
                        dynamics = json.load(handle)
                    supplied = getattr(network, "_generated_basis_weights", {}).get(
                        filename
                    )
                    if supplied is None:
                        supplied = (network._basis_weights or {}).get(filename.stem)
                    weights = (
                        supplied
                        if supplied is not None
                        else dynamics.get("basis_weights")
                    )
                    records[filename] = {
                        "dynamics": dynamics,
                        "weights": weights,
                        "uses": [],
                    }
                records[filename]["uses"].append((population, type_id))
    missing = {
        filename: record
        for filename, record in records.items()
        if force_recompute or record["weights"] is None
    }
    if not missing:
        if force_recompute:
            raise ValueError(
                "force_recompute requires SONATA synaptic dynamics files with raw kinetics"
            )
        return None
    if not force_recompute and tau_basis is None and len(missing) != len(records):
        raise ValueError(
            "Partially supplied basis weights require their shared tau_basis; refusing to refit supplied coefficients"
        )
    targets = {network.name: network for network in networks}
    target_types_cache = {}
    parameters = []
    filenames = []
    for filename, record in missing.items():
        dynamics = record["dynamics"]
        if "tau_syn" in dynamics or "tau_syn_fast" in dynamics:
            fast = dynamics.get("tau_syn_fast", dynamics.get("tau_syn"))
            slow = dynamics.get("tau_syn_slow", fast)
            candidates = [
                np.array([fast, slow, dynamics.get("amp_slow", 0.0)], dtype=float)
            ]
        elif "receptor_type" in dynamics:
            receptor = dynamics["receptor_type"]
            if not isinstance(receptor, int) or receptor < 1:
                raise ValueError(
                    "receptor_type must be a positive one-based integer: {}".format(
                        filename
                    )
                )
            candidates = []
            for population, type_id in record["uses"]:
                target = targets[population.target_population]
                if id(population) not in target_types_cache:
                    target_types_cache[id(population)] = _target_types_by_edge(
                        population, target
                    )
                node_types = target_types_cache[id(population)][type_id]
                lookup = target._dynamics_params_lu
                for node_type in node_types:
                    index = lookup["node_type_id"].index(node_type)
                    try:
                        candidates.append(
                            np.array(
                                [
                                    lookup[name][index][receptor - 1]
                                    for name in (
                                        "tau_syn_fast",
                                        "tau_syn_slow",
                                        "amp_slow",
                                    )
                                ]
                            )
                        )
                    except (KeyError, IndexError, TypeError) as error:
                        raise ValueError(
                            "Missing receptor kinetics for {} / target type {}".format(
                                filename, node_type
                            )
                        ) from error
        else:
            raise ValueError(
                "Missing raw synaptic kinetics for {}; provide tau_syn[_fast], tau_syn_slow and amp_slow, or receptor_type with target-cell kinetics".format(
                    filename
                )
            )
        if not candidates or any(
            not np.array_equal(candidate, candidates[0]) for candidate in candidates[1:]
        ):
            raise ValueError(
                "{} maps to multiple target kinetics; use distinct synaptic dynamics files for those classes".format(
                    filename
                )
            )
        parameters.append(candidates[0])
        filenames.append(filename)
    result = fit_alpha_basis(parameters, tau_basis=tau_basis, **options)
    for network in networks:
        if hasattr(network, "_basis_weights"):
            generated = dict(getattr(network, "_generated_basis_weights", {}))
            for filename, weights in zip(filenames, result["weights"]):
                generated[filename] = weights
            network._generated_basis_weights = generated
    cell_params["tau_basis"] = result["tau_basis"]
    if force_recompute:
        cell_params.pop("synaptic_basis_weights", None)
    result["diagnostics"]["force_recompute"] = force_recompute
    result["diagnostics"]["synaptic_files"] = [str(filename) for filename in filenames]
    return result
