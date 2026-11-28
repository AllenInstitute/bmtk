from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions.weight_regularization import (
    _normalize_group_by, attribute_group_keys,
)


class _Adaptor:
    def __init__(self, nodes, edge_types):
        self.nodes = nodes
        self.n_nodes = len(next(iter(nodes.values())))
        table = SimpleNamespace(to_dataframe=lambda: edge_types)
        self._sonata_edge_pops = [SimpleNamespace(types_table=table)]

    def node_attribute_values(self, columns):
        return {column: self.nodes[column] for column in columns}


def test_attribute_group_keys_match_tuple_grouping():
    rng = np.random.default_rng(0)
    nodes = {"area": rng.choice(["V1", "LM", "AL"], 30), "cls": rng.choice(["E", "PV", "SST"], 30)}
    edge_types = pd.DataFrame({"receptor": [1, 2, 1, 4]}, index=pd.Index([100, 101, 102, 103], name="edge_type_id"))
    adaptor = _Adaptor(nodes, edge_types)
    group_by = _normalize_group_by({"target": ["area", "cls"], "source": ["cls"], "edge": ["receptor"]})
    targets, sources = rng.integers(0, 30, 500), rng.integers(0, 30, 500)
    types = rng.choice([100, 101, 102, 103], 500)
    keys = attribute_group_keys(adaptor, group_by, {"a": (targets[:300], sources[:300], types[:300]),
                                                    "b": (targets[300:], sources[300:], types[300:])})
    key = np.concatenate([keys["a"], keys["b"]])
    receptor = edge_types["receptor"].reindex(types).to_numpy()
    tuples = list(zip(nodes["area"][targets], nodes["cls"][targets], nodes["cls"][sources], receptor))
    for i in range(500):
        for j in range(i + 1, min(i + 40, 500)):
            assert (key[i] == key[j]) == (tuples[i] == tuples[j])
    assert len(np.unique(key)) == len(set(tuples))


@pytest.mark.parametrize("group_by", [{"target": []}, {"tgt": ["area"]}, ["area"]])
def test_group_by_validation(group_by):
    with pytest.raises(ValueError, match="group_by"):
        _normalize_group_by(group_by)


def test_node_group_columns_follow_permuted_node_group_index(tmp_path):
    h5py = pytest.importorskip("h5py")
    from bmtk.utils.sonata import File
    from bmtk.simulator.dpointnet.network_adaptor import SONATANetwork

    nodes, types = tmp_path / "nodes.h5", tmp_path / "node_types.csv"
    with h5py.File(nodes, "w") as f:
        f.attrs["magic"] = np.uint32(0x0A7A)
        f.attrs["version"] = np.array([0, 1], np.uint32)
        pop = f.create_group("nodes/cortex")
        pop["node_id"] = np.arange(4)
        pop["node_type_id"] = np.full(4, 100)
        pop["node_group_id"] = np.zeros(4, np.int64)
        pop["node_group_index"] = np.array([2, 0, 3, 1])
        pop["0/area"] = np.array([b"V1", b"LM", b"AL", b"RL"])
    types.write_text("node_type_id cell_class\n100 E\n")
    population = File(data_files=str(nodes), data_type_files=str(types)).nodes["cortex"]
    adaptor = object.__new__(SONATANetwork)
    adaptor._sonata_node_pop = population
    values = adaptor.node_attribute_values(["area", "cell_class"])
    area = [v.decode() if isinstance(v, bytes) else v for v in values["area"]]
    assert area == ["AL", "V1", "RL", "LM"]
    assert list(values["cell_class"]) == ["E"] * 4



def test_node_attribute_values_span_multiple_node_groups(tmp_path):
    h5py = pytest.importorskip("h5py")
    from bmtk.utils.sonata import File
    from bmtk.simulator.dpointnet.network_adaptor import SONATANetwork

    nodes, types = tmp_path / "nodes.h5", tmp_path / "node_types.csv"
    with h5py.File(nodes, "w") as f:
        f.attrs["magic"] = np.uint32(0x0A7A)
        f.attrs["version"] = np.array([0, 1], np.uint32)
        pop = f.create_group("nodes/v1")
        pop["node_id"] = np.arange(5)
        pop["node_type_id"] = np.array([100, 101, 100, 101, 100])
        pop["node_group_id"] = np.array([0, 1, 0, 1, 0])
        pop["node_group_index"] = np.array([2, 1, 0, 0, 1])
        pop["0/x"] = np.array([10.0, 11.0, 12.0])
        pop["0/z"] = np.array([0.0, 1.0, 2.0])
        pop["1/x"] = np.array([20.0, 21.0])
        pop["1/z"] = np.array([5.0, 6.0])
        pop["1/tuning"] = np.array([0.5, 0.6])
    types.write_text("node_type_id pop_name\n100 e4\n101 i4\n")
    population = File(data_files=str(nodes), data_type_files=str(types)).nodes["v1"]
    adaptor = object.__new__(SONATANetwork)
    adaptor._sonata_node_pop = population
    values = adaptor.node_attribute_values(["x", "z", "pop_name"])
    np.testing.assert_array_equal(values["x"], [12.0, 21.0, 10.0, 20.0, 11.0])
    np.testing.assert_array_equal(values["z"], [2.0, 6.0, 0.0, 5.0, 1.0])
    assert list(values["pop_name"]) == ["e4", "i4", "e4", "i4", "e4"]
    with pytest.raises(ValueError, match="not found in node group 0"):
        adaptor.node_attribute_values(["tuning"])
