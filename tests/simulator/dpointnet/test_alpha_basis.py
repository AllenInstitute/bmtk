import numpy as np
import pytest
import json
from types import SimpleNamespace
import h5py
import pandas as pd

from bmtk.simulator.dpointnet.alpha_basis import (
    alpha_design,
    fit_alpha_basis,
    prepare_alpha_basis,
)


def test_fixed_basis_recovers_double_alpha_coefficients():
    result = fit_alpha_basis(
        [[1.0, 4.0, 0.3], [4.0, 1.0, 0.2]], 1e-10, tau_basis=[1.0, 4.0]
    )
    np.testing.assert_allclose(result["weights"], [[1.0, 0.3], [0.2, 1.0]], atol=1e-12)
    assert result["relative_rms"].max() < 1e-12


def test_optimized_shared_basis_and_determinism():
    parameters = [[1.0, 4.0, 0.3], [2.0, 8.0, 0.2]]
    options = dict(tolerance=0.01, n_points=200, max_iterations=15)
    first = fit_alpha_basis(parameters, **options)
    second = fit_alpha_basis(parameters, **options)
    np.testing.assert_array_equal(first["tau_basis"], second["tau_basis"])
    assert len(first["tau_basis"]) == 4
    assert first["relative_rms"].max() <= 0.01
    assert first["diagnostics"]["elapsed_seconds"] > 0


def test_unattainable_tolerance_fails():
    with pytest.raises(ValueError, match="did not meet"):
        fit_alpha_basis([[1.0, 4.0, 0.3]], 1e-12, tau_basis=[2.0])


@pytest.mark.parametrize(
    "options",
    [
        {"min_basis": 2.5},
        {"max_basis": 3},
        {"n_points": 5},
        {"tolerance": 0},
        {"seed": -1},
        {"max_iterations": 0},
    ],
)
def test_invalid_fit_options(options):
    with pytest.raises(ValueError):
        fit_alpha_basis([[1, 4, 0.3]], **options)


@pytest.mark.parametrize("parameters", [[], [[0, 1, 0]], [[1, np.nan, 0]], [[1, 2]]])
def test_invalid_kinetics(parameters):
    with pytest.raises(ValueError, match="parameters"):
        fit_alpha_basis(parameters, 0.01)


def test_peak_normalization():
    np.testing.assert_allclose(np.diag(alpha_design([1.0, 4.0], [1.0, 4.0])), 1.0)


def synaptic_network(tmp_path, dynamics):
    filename = tmp_path / "synapse.json"
    filename.write_text(json.dumps(dynamics))
    population = SimpleNamespace(
        type_ids=np.array([1]), types_table={1: {"dynamics_params": filename}}
    )
    network = SimpleNamespace(
        name="v1", _sonata_edge_pops=[population], _basis_weights=None
    )
    return network, filename


def test_prepare_missing_coefficients_keeps_input_files_immutable(tmp_path):
    network, filename = synaptic_network(
        tmp_path, {"tau_syn": 1.0, "tau_syn_slow": 4.0, "amp_slow": 0.3}
    )
    original = filename.read_bytes()
    params = {"tau_basis": [1.0, 4.0]}
    result = prepare_alpha_basis([network], params, {"tolerance": 1e-10})
    np.testing.assert_allclose(
        network._generated_basis_weights[filename], [1.0, 0.3], atol=1e-12
    )
    assert filename.read_bytes() == original
    assert result["diagnostics"]["synaptic_files"] == [str(filename)]
    assert prepare_alpha_basis([network], params) is None


def test_precomputed_coefficients_do_not_trigger_fitting(tmp_path):
    network, _ = synaptic_network(tmp_path, {"basis_weights": [1, 0, 0, 0, 0]})
    assert prepare_alpha_basis([network], {"tau_basis": [1, 2, 3, 4, 5]}) is None
    assert network._basis_weights is None


@pytest.mark.parametrize("force_recompute", [None, "true", 1])
def test_force_recompute_requires_boolean(force_recompute):
    with pytest.raises(ValueError, match="force_recompute must be"):
        prepare_alpha_basis([], {}, {"force_recompute": force_recompute})


def test_force_recompute_failure_preserves_existing_coefficients(tmp_path):
    network, filename = synaptic_network(tmp_path, {"basis_weights": [99, 99]})
    original = filename.read_bytes()
    params = {"tau_basis": [1, 4], "synaptic_basis_weights": [[99, 99]]}
    with pytest.raises(ValueError, match="Missing raw synaptic kinetics"):
        prepare_alpha_basis([network], params, {"force_recompute": True})
    assert params == {"tau_basis": [1, 4], "synaptic_basis_weights": [[99, 99]]}
    assert filename.read_bytes() == original
    assert not hasattr(network, "_generated_basis_weights")


def test_force_recompute_requires_sonata_inputs():
    with pytest.raises(ValueError, match="requires SONATA"):
        prepare_alpha_basis([], {}, {"force_recompute": True})


def test_missing_kinetics_fail_clearly(tmp_path):
    network, _ = synaptic_network(tmp_path, {})
    with pytest.raises(ValueError, match="Missing raw synaptic kinetics"):
        prepare_alpha_basis([network], {}, {"tolerance": 0.01})
    assert prepare_alpha_basis([network], {}, False) is None


def test_auto_selects_five_for_broad_kinetics():
    taus = np.array([0.2, 0.7, 2.0, 7.0, 24.0])
    parameters = np.column_stack([taus, taus, np.zeros(5)])
    result = fit_alpha_basis(parameters, tolerance=0.01, n_points=1000)
    assert len(result["tau_basis"]) == 5
    assert result["diagnostics"]["attempts"][0]["max_relative_rms"] > 0.01
    assert result["relative_rms"].max() <= 0.01


def test_receptor_kinetics_and_noncanonical_node_ids(tmp_path):
    network, filename = synaptic_network(tmp_path, {"receptor_type": 2})
    population = network._sonata_edge_pops[0]
    population.target_population = "v1"
    population._type_id_ds = np.array([1, 1])
    population._target_node_id_ds = np.array([17, 3])
    network._sonata_node_pop = SimpleNamespace(
        node_ids=np.array([17, 3]), type_ids=np.array([10, 10])
    )
    network._dynamics_params_lu = {
        "node_type_id": [10],
        "tau_syn_fast": [[2.0, 1.0]],
        "tau_syn_slow": [[8.0, 4.0]],
        "amp_slow": [[0.5, 0.3]],
    }
    result = prepare_alpha_basis(
        [network], {"tau_basis": [1.0, 4.0]}, {"tolerance": 1e-10}
    )
    np.testing.assert_allclose(result["weights"], [[1.0, 0.3]], atol=1e-12)
    np.testing.assert_allclose(
        network._generated_basis_weights[filename], [1.0, 0.3], atol=1e-12
    )


def sonata_config(tmp_path, tau_basis=None, csv_weights=None):
    from bmtk.simulator.core.simulation_config import SimulationConfig

    cell = {
        "V_th": -50.0,
        "g": 10.0,
        "C_m": 100.0,
        "E_L": -70.0,
        "V_reset": -70.0,
        "t_ref": 2.0,
        "asc_decay": [0.01, 0.1],
        "asc_amps": [0.0, 0.0],
    }
    (tmp_path / "cell.json").write_text(json.dumps(cell))
    for name, fast, slow in (("rec", 1.0, 4.0), ("drive", 2.0, 8.0)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"tau_syn": fast, "tau_syn_slow": slow, "amp_slow": 0.3})
        )
    nodes_specs = []
    for name, node_type, model_type in (
        ("v1", 10, "point_neuron"),
        ("drive", 20, "virtual"),
    ):
        filename = tmp_path / f"{name}_nodes.h5"
        with h5py.File(filename, "w") as handle:
            group = handle.create_group(f"nodes/{name}")
            for key, values in (
                ("node_id", [0]),
                ("node_type_id", [node_type]),
                ("node_group_id", [0]),
                ("node_group_index", [0]),
            ):
                group[key] = values
            group.create_group("0")
        types = {"node_type_id": [node_type], "model_type": [model_type]}
        if name == "v1":
            types["dynamics_params"] = ["cell.json"]
        type_file = tmp_path / f"{name}_node_types.csv"
        pd.DataFrame(types).to_csv(type_file, sep=" ", index=False)
        nodes_specs.append(
            {"nodes_file": str(filename), "node_types_file": str(type_file)}
        )
    edges_specs = []
    for source, synapse in (("v1", "rec"), ("drive", "drive")):
        filename = tmp_path / f"{source}_edges.h5"
        with h5py.File(filename, "w") as handle:
            group = handle.create_group(f"edges/{source}_to_v1")
            for key, values in (
                ("source_node_id", [0]),
                ("target_node_id", [0]),
                ("edge_type_id", [100]),
                ("edge_group_id", [0]),
                ("edge_group_index", [0]),
            ):
                group[key] = values
            group["source_node_id"].attrs["node_population"] = source
            group["target_node_id"].attrs["node_population"] = "v1"
            group["0/syn_weight"] = [0.0 if source == "v1" else 10.0]
        type_file = tmp_path / f"{source}_edge_types.csv"
        pd.DataFrame(
            {
                "edge_type_id": [100],
                "dynamics_params": [f"{synapse}.json"],
                "delay": [1.0],
            }
        ).to_csv(type_file, sep=" ", index=False)
        edges_specs.append(
            {"edges_file": str(filename), "edge_types_file": str(type_file)}
        )
    params = {
        "cell_model": "GLIF3",
        "use_fused_cuda": False,
        "use_fused_state": False,
        "alpha_basis": {"n_points": 200, "max_iterations": 15},
    }
    if tau_basis is not None:
        params["tau_basis"] = tau_basis
    components = {
        "point_neuron_models_dir": str(tmp_path),
        "synaptic_models_dir": str(tmp_path),
    }
    if csv_weights is not None:
        filename = tmp_path / "basis.csv"
        pd.DataFrame(csv_weights).to_csv(filename, index=False)
        components["basis_weights_file"] = str(filename)
    return SimulationConfig(
        {
            "run": {"seq_len": 4, "batch_size": 1, "dt": 1.0, "dtype": "float32"},
            "rnn_cell_params": params,
            "components": components,
            "networks": {"nodes": nodes_specs, "edges": edges_specs},
            "inputs": {},
        }
    )


@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("acceleration_profile", [None, "auto"])
def test_real_sonata_rnn_uses_shared_basis_for_recurrent_and_input(
    tmp_path, provided, acceleration_profile, monkeypatch
):
    from bmtk.simulator.dpointnet.rnn_model import RNN
    from bmtk.simulator.dpointnet.id_maps import TFIDMap

    monkeypatch.setattr(TFIDMap, "_tf_id_map_instance", None)

    columns = {"connection_name": ["rec", "drive"], "note": ["keep", "keep"]}
    columns.update(
        {
            f"w{index}": [1.0 if index == 0 else 0.0, 1.0 if index == 1 else 0.0]
            for index in range(5)
        }
    )
    config = sonata_config(
        tmp_path, [1, 2, 4, 8, 16] if provided else None, columns if provided else None
    )
    rnn = RNN.from_config(config)
    rnn.cell_params["acceleration_profile"] = acceleration_profile
    try:
        rnn.build()
        assert (rnn.acceleration_report is None) == (acceleration_profile is None)
        assert rnn.cell.synaptic_basis_weights.shape == (2, 5 if provided else 4)
        assert (rnn.alpha_basis_fit is None) == provided
        if provided:
            np.testing.assert_array_equal(
                rnn.cell.synaptic_basis_weights, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
            )
        else:
            assert len(rnn.cell_params["tau_basis"]) == 4
            assert len(rnn.alpha_basis_fit["diagnostics"]["synaptic_files"]) == 2
        outputs, final_state = rnn.cell(
            np.ones((1, 1), dtype=np.float32), rnn.cell.zero_state(1, rnn.dtype)
        )
        assert all(np.isfinite(value.numpy()).all() for value in final_state)
    finally:
        rnn.cleanup()


@pytest.mark.parametrize("supplied", ["csv", "embedded", "explicit"])
def test_force_recompute_replaces_entire_shared_basis_without_input_writes(
    tmp_path, supplied, monkeypatch
):
    from bmtk.simulator.dpointnet.rnn_model import RNN
    from bmtk.simulator.dpointnet.id_maps import TFIDMap

    monkeypatch.setattr(TFIDMap, "_tf_id_map_instance", None)
    columns = {"connection_name": ["rec", "drive"]}
    columns.update({f"w{index}": [99.0, 99.0] for index in range(5)})
    config = sonata_config(
        tmp_path, [20, 30, 40, 50, 60], columns if supplied == "csv" else None
    )
    if supplied == "embedded":
        for name in ("rec", "drive"):
            filename = tmp_path / f"{name}.json"
            parameters = json.loads(filename.read_text())
            parameters["basis_weights"] = [99.0] * 5
            filename.write_text(json.dumps(parameters))
    elif supplied == "explicit":
        config["rnn_cell_params"]["synaptic_basis_weights"] = [[99.0] * 5] * 2
    config["rnn_cell_params"]["alpha_basis"]["force_recompute"] = True
    originals = {filename: filename.read_bytes() for filename in tmp_path.iterdir()}
    rnn = RNN.from_config(config)
    try:
        rnn.build()
        assert rnn.alpha_basis_fit["diagnostics"]["force_recompute"] is True
        assert len(rnn.cell_params["tau_basis"]) == 4
        assert "synaptic_basis_weights" not in rnn.cell_params
        np.testing.assert_allclose(
            rnn.cell.synaptic_basis_weights, rnn.alpha_basis_fit["weights"], rtol=1e-6
        )
        assert rnn.cell.synaptic_basis_weights.shape == (2, 4)
        assert rnn.alpha_basis_fit["relative_rms"].max() <= 0.08012288897995931
        assert all(
            filename.read_bytes() == content for filename, content in originals.items()
        )
        previous_taus = rnn.cell_params["tau_basis"].copy()
        rnn.build(rebuild=True)
        np.testing.assert_array_equal(rnn.cell_params["tau_basis"], previous_taus)
        assert rnn.alpha_basis_fit["diagnostics"]["force_recompute"] is True
    finally:
        rnn.cleanup()
