from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from bmtk.simulator.dpointnet.network_adaptor import NetworkAdaptor, SONATANetwork
from bmtk.simulator.dpointnet.weights import ModelWeights


@pytest.mark.parametrize("recurrent", [True, False])
def test_export_preserves_order_physical_weights_and_input_scaling(tmp_path, recurrent):
    adaptor = SimpleNamespace(
        canonical_edge_order=np.array([1, 2, 0]) if recurrent else None,
        connection_table=pd.DataFrame(
            {
                "source_node_id": [2, 0, 1],
                "target_node_id": [0, 1, 0],
                "source_population": ["v1" if recurrent else "bkg"] * 3,
                "target_population": ["v1"] * 3,
                "edge_type_id": [12, 10, 11],
            }
        ),
    )
    cell = SimpleNamespace(
        recurrent_weight_values=np.array([2.0, 3.0, 5.0]),
        _recurrent_export_factor=np.array([10.0, 20.0, 30.0]),
        inputs={
            "bkg": {
                "input_weight_values": np.array([7.0, 11.0, 13.0]),
                "export_factor": np.array([2.0, 3.0, 4.0]),
            }
        },
    )
    rnn = SimpleNamespace(inputs_populations=["bkg"], get_network=lambda name: adaptor)
    weights = ModelWeights(rnn, cell)
    weights._network_weights = {
        ("<recurrent>" if recurrent else "bkg"): weights._network_weights[
            "<recurrent>" if recurrent else "bkg"
        ]
    }
    expected_order = [1, 2, 0] if recurrent else [0, 1, 2]
    expected_weights = (
        np.array([20.0, 60.0, 150.0]) if recurrent else np.array([14.0, 33.0, 52.0])
    )
    weights.to_sonata(tmp_path)
    source = "v1" if recurrent else "bkg"
    with h5py.File(tmp_path / f"{source}_v1_edges.h5") as handle:
        group = handle[f"edges/{source}_to_v1"]
        for name in ("source_node_id", "target_node_id", "edge_type_id"):
            np.testing.assert_array_equal(
                group[name], adaptor.connection_table[name].to_numpy()[expected_order]
            )
        np.testing.assert_array_equal(
            group["0/syn_weight"], expected_weights[expected_order]
        )
    with pytest.raises(ValueError, match="already exists"):
        weights.to_sonata(tmp_path)
    weights.to_sonata(tmp_path, overwrite=True)


def test_group_row_mapping_handles_interleaved_and_empty_groups():
    population = SimpleNamespace(
        group_indicies=lambda group_id: (
            (3, [(0, 1), (2, 4)]) if group_id == 1 else (0, [])
        ),
        _group_index_ds=np.array([2, 7, 0, 1]),
    )
    rows, group_indices = SONATANetwork._edge_group_rows(
        population, SimpleNamespace(group_id=1)
    )
    np.testing.assert_array_equal(rows, [0, 2, 3])
    np.testing.assert_array_equal(group_indices, [2, 0, 1])
    rows, group_indices = SONATANetwork._edge_group_rows(
        population, SimpleNamespace(group_id=2)
    )
    assert rows.size == group_indices.size == 0
    assert NetworkAdaptor("test", "input").canonical_edge_order is None


@pytest.mark.parametrize("network_type", ["recurrent", "input"])
def test_sonata_loader_preserves_interleaved_groups_and_population_offsets(
    tmp_path, network_type
):
    from bmtk.utils import sonata

    types_path = tmp_path / "types.csv"
    pd.DataFrame(
        {
            "edge_type_id": [10, 11],
            "dynamics_params": ["unused", "unused"],
            "delay": [1.0, 2.0],
        }
    ).to_csv(types_path, sep=" ", index=False)
    filename = tmp_path / "edges.h5"
    with h5py.File(filename, "w") as handle:
        for name, source, target, type_ids, groups, indices in (
            ("first", [2, 0, 1], [0, 1, 0], [10, 11, 10], [1, 0, 1], [1, 0, 0]),
            ("second", [1], [2], [11], [0], [0]),
        ):
            edges = handle.create_group(f"edges/{name}")
            edges["source_node_id"] = source
            edges["source_node_id"].attrs["node_population"] = "v1"
            edges["target_node_id"] = target
            edges["target_node_id"].attrs["node_population"] = "v1"
            edges["edge_type_id"] = type_ids
            edges["edge_group_id"] = groups
            edges["edge_group_index"] = indices
            edges["0/syn_weight"] = [17.0] if name == "first" else [23.0]
            if name == "first":
                edges["1/syn_weight"] = [11.0, 13.0]
    sonata_file = sonata.File(data_files=str(filename), data_type_files=str(types_path))
    adaptor = object.__new__(SONATANetwork)
    NetworkAdaptor.__init__(adaptor, "v1", network_type)
    adaptor._sonata_node_pop = SimpleNamespace(
        node_ids=np.arange(3),
        type_ids=np.ones(3, dtype=int),
        to_dataframe=lambda: pd.DataFrame({"node_id": np.arange(3)}),
    )
    adaptor._sonata_edge_pops = [
        sonata_file.edges["first"],
        sonata_file.edges["second"],
    ]
    adaptor.id_maps = SimpleNamespace(bmtk2tf_id_map=lambda population: np.arange(3))
    adaptor._cache_file = None
    adaptor._dynamics_params_lu = {
        key: [1.0]
        for key in (
            "V_th",
            "g",
            "E_L",
            "asc_decay",
            "C_m",
            "V_reset",
            "t_ref",
            "asc_amps",
        )
    }
    adaptor._dynamics_params_lu["node_type_id"] = [1]
    lookup = pd.DataFrame({"dyn_params_idx": [0, 1]}, index=[10, 11])
    adaptor._synaptic_dyn_params = lambda: ({}, {"first": lookup, "second": lookup})
    loaded = adaptor.to_dict()
    synapses = loaded["synapses"] if network_type == "recurrent" else loaded
    order = (
        adaptor.canonical_edge_order if network_type == "recurrent" else np.arange(4)
    )
    np.testing.assert_array_equal(
        synapses["indices"][order], [[0, 2], [1, 0], [0, 1], [2, 1]]
    )
    np.testing.assert_array_equal(synapses["weights"][order], [13.0, 17.0, 11.0, 23.0])
    np.testing.assert_array_equal(synapses["delays"][order], [1.0, 2.0, 1.0, 2.0])
    np.testing.assert_array_equal(adaptor._edge_type_ids[order], [10, 11, 10, 11])
