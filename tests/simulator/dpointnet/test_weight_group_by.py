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
