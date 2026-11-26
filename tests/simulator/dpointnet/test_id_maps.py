import numpy as np
import pytest

from bmtk.simulator.dpointnet.id_maps import TFIDMap


@pytest.fixture
def id_map(monkeypatch):
    monkeypatch.setattr(TFIDMap, "_tf_id_map_instance", None)
    monkeypatch.setattr(TFIDMap, "_initialized", False)
    return TFIDMap()


@pytest.mark.parametrize("node_ids", [
    [0, 1, 2, 3],
    [3, 0, 2, 1],           # permutation (node file order differs from node IDs)
    [10, 4, 7, 100],        # gaps and out of order
])
def test_tf2bmtk_inverts_the_node_map(id_map, node_ids):
    id_map.add_bmtk_ids("cells", np.array(node_ids), "recurrent")
    table = id_map.tf2bmtk_id_map()
    np.testing.assert_array_equal(table.index.to_numpy(), np.arange(len(node_ids)))
    np.testing.assert_array_equal(table["node_id"].to_numpy(), node_ids)
    forward = id_map.bmtk2tf_id_map("cells")
    np.testing.assert_array_equal(forward[table["node_id"].to_numpy()], table.index.to_numpy())


def test_tf2bmtk_offsets_later_recurrent_populations(id_map):
    id_map.add_bmtk_ids("a", np.array([5, 1]), "recurrent")
    id_map.add_bmtk_ids("b", np.array([2, 0, 9]), "recurrent")
    table = id_map.tf2bmtk_id_map()
    np.testing.assert_array_equal(table.index.to_numpy(), np.arange(5))
    np.testing.assert_array_equal(table["node_id"].to_numpy(), [5, 1, 2, 0, 9])
    np.testing.assert_array_equal(table["population"].to_numpy(), ["a", "a", "b", "b", "b"])
