"""Neuron-partition planning, rank-local SONATA loading and the MPI transport oracle."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from bmtk.simulator.dpointnet.neuron_partition import (
    PARTITION_INPUT, NeuronPartition, neuron_partition_requested, plan_partition,
)

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "dpointnet_all2all"


def test_contiguous_plan_balances_edges_and_gives_every_rank_neurons():
    in_degree = np.r_[np.full(10, 100), np.zeros(30, int), np.full(10, 100)]
    owner = plan_partition(50, 2, in_degree=in_degree)
    assert np.all(np.diff(owner) >= 0) and set(owner) == {0, 1}
    loads = np.bincount(owner, weights=in_degree)
    assert abs(loads[0] - loads[1]) <= 100
    np.testing.assert_array_equal(plan_partition(50, 2, in_degree=in_degree), owner)
    assert np.bincount(plan_partition(5, 5, balance="neurons")).tolist() == [1] * 5
    owner = plan_partition(4, 4, in_degree=[1000, 0, 0, 0])
    assert owner.tolist() == [0, 1, 2, 3]


def test_grouped_plan_keeps_groups_whole():
    groups = np.repeat(["a", "b", "c", "d", "e"], [7, 3, 9, 2, 4])
    owner = plan_partition(len(groups), 3, in_degree=np.ones(len(groups)), groups=groups)
    for label in np.unique(groups):
        assert len(set(owner[groups == label])) == 1
    assert set(owner) == {0, 1, 2}


@pytest.mark.parametrize("arguments,error", [
    (dict(n_nodes=2, size=3, balance="neurons"), "Cannot split"),
    (dict(n_nodes=4, size=2), "in-degrees"),
    (dict(n_nodes=4, size=2, balance="sizes"), "Unknown partition balance"),
    (dict(n_nodes=4, size=3, balance="neurons", groups=[0, 0, 1, 1]), "groups cannot occupy"),
])
def test_plan_rejects_invalid_requests(arguments, error):
    with pytest.raises(ValueError, match=error):
        plan_partition(**arguments)


def test_partition_request_follows_launch_and_strategy():
    assert neuron_partition_requested({"strategy": "neurons"}, 1, False)
    assert neuron_partition_requested(None, 4, False)
    assert not neuron_partition_requested(None, 1, False)
    assert not neuron_partition_requested(None, 4, True)
    assert not neuron_partition_requested({"strategy": "mpi"}, 4, True)


def test_neuron_partition_positions():
    partition = NeuronPartition(np.array([1, 0, 1, 0, 2]), 1)
    assert partition.local_nodes.tolist() == [0, 2] and partition.n_local == 2
    assert partition.position.tolist() == [0, 0, 1, 1, 0]
    assert partition.is_local([0, 1, 2]).tolist() == [True, False, True]


@pytest.fixture(scope="module")
def example_networks():
    if not (EXAMPLE / "network" / "glifs_glifs_edges.h5").exists():
        pytest.skip("dpointnet_all2all example network not available")
    from bmtk.simulator.dpointnet.config import Config
    from bmtk.simulator.dpointnet.rnn_model import RNN

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        rnn = RNN.from_config(Config.from_json("config.train.json"))
        recurrent = next(iter(rnn._recurrent_networks.values()))
        inputs = list(rnn._input_networks.values())
        for network in [recurrent, *inputs]:
            network._synaptic_dyn_params()
        reference = recurrent.to_dict()
        input_references = {network.name: network.to_dict() for network in inputs}
        yield recurrent, inputs, reference, input_references
    finally:
        os.chdir(cwd)


def test_chunked_reader_equals_whole_network_builders(example_networks, monkeypatch):
    recurrent, inputs, reference, input_references = example_networks
    monkeypatch.setattr(type(recurrent), "_EDGE_CHUNK_ROWS", 997)
    edges = recurrent.read_edges()
    order = np.lexsort((edges["rows"], edges["source"], edges["target"]))
    synapses = reference["synapses"]
    np.testing.assert_array_equal(synapses["indices"][:, 0], edges["target"][order])
    np.testing.assert_array_equal(synapses["indices"][:, 1], edges["source"][order])
    for name in ("weights", "delays", "syn_ids"):
        np.testing.assert_array_equal(synapses[name], edges[name][order])
    degree = recurrent.target_in_degree()
    np.testing.assert_array_equal(degree, np.bincount(edges["target"], minlength=recurrent.n_nodes))
    np.testing.assert_array_equal(sum(recurrent.target_in_degree((i, 3)) for i in range(3)), degree)
    for network in inputs:
        edges, ref = network.read_edges(), input_references[network.name]
        np.testing.assert_array_equal(ref["indices"], np.column_stack((edges["target"], edges["source"])))
        for name in ("weights", "delays", "syn_ids"):
            np.testing.assert_array_equal(np.asarray(ref[name]), edges[name])


@pytest.mark.parametrize("size", [2, 3])
def test_rank_dicts_cover_the_network_exactly(example_networks, size):
    recurrent, inputs, reference, input_references = example_networks
    synapses = reference["synapses"]
    n_edges = len(synapses["weights"])
    degree = recurrent.target_in_degree()
    owner = plan_partition(recurrent.n_nodes, size, in_degree=degree)
    offset = np.r_[0, np.cumsum(degree)][:-1]
    seen = np.zeros(n_edges, dtype=int)
    input_seen = {network.name: np.zeros(len(input_references[network.name]["weights"]), int) for network in inputs}
    for rank in range(size):
        partition = NeuronPartition(owner, rank)
        local, remote, canonical = recurrent.partition_recurrent_dicts(
            partition, offset, {"trainable": True, "weight_scale": 1.0})
        assert local["n_nodes"] == partition.n_local
        np.testing.assert_array_equal(local["node_type_ids"], reference["node_type_ids"][partition.local_nodes])
        sources = remote["remote_sources"]
        assert remote["n_inputs"] == sources.size and not partition.is_local(sources).any()
        for name, table, source_ids in (
            ("<recurrent>", local["synapses"], partition.local_nodes),
            (PARTITION_INPUT, remote, sources),
        ):
            ids = canonical[name]
            seen[ids] += 1
            np.testing.assert_array_equal(partition.local_nodes[table["indices"][:, 0]], synapses["indices"][ids, 0])
            np.testing.assert_array_equal(source_ids[table["indices"][:, 1]], synapses["indices"][ids, 1])
            for field in ("weights", "delays", "syn_ids"):
                np.testing.assert_array_equal(table[field], synapses[field][ids])
            assert np.all(np.diff(ids) > 0), "rank edges must keep the global canonical order"
        for network in inputs:
            table, rows = network.partition_input_dict(partition)
            ref = input_references[network.name]
            input_seen[network.name][rows] += 1
            np.testing.assert_array_equal(partition.local_nodes[table["indices"][:, 0]], ref["indices"][rows, 0])
            np.testing.assert_array_equal(table["indices"][:, 1], ref["indices"][rows, 1])
            np.testing.assert_array_equal(table["weights"], np.asarray(ref["weights"])[rows])
    assert np.all(seen == 1)
    assert all(np.all(count == 1) for count in input_seen.values())


@pytest.mark.skipif(
    os.environ.get("BMTK_RUN_MPI_TESTS") != "1",
    reason="Real MPI subprocess tests require explicit BMTK_RUN_MPI_TESTS=1.",
)
@pytest.mark.parametrize("ranks", [2, 3])
def test_mpi_transport_collectives_and_grouped_emd(ranks):
    pytest.importorskip("mpi4py")
    launcher = shutil.which("mpiexec")
    assert launcher, "MPI tests require mpiexec."
    probe = Path(__file__).with_name("neuron_partition_probe.py")
    command = [launcher, "-n", str(ranks), sys.executable, str(probe)]
    if "Open MPI" in subprocess.run([launcher, "--version"], capture_output=True, text=True).stdout:
        command.insert(1, "--oversubscribe")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"neuron partition probe passed on {ranks} ranks" in result.stdout


@pytest.mark.skipif(
    os.environ.get("BMTK_RUN_MPI_TESTS") != "1",
    reason="Real MPI subprocess tests require explicit BMTK_RUN_MPI_TESTS=1.",
)
def test_partitioned_training_matches_single_process_on_gpu(tmp_path):
    """The example trained under mpiexec -n 2 equals one process: losses and exported weights."""
    pytest.importorskip("mpi4py")
    import h5py
    import pandas as pd
    from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available

    if not fused_cuda_available():
        pytest.skip("Neuron-partitioned training uses the GPU direct recorded adjoint.")
    launcher = shutil.which("mpiexec")
    assert launcher, "MPI tests require mpiexec."
    example = Path(__file__).resolve().parents[3] / "examples" / "dpointnet_mpi"
    config, script = example / "config.partition.json", example / "run_training.py"
    outputs = {}
    for ranks in (1, 2):
        output = tmp_path / f"ranks{ranks}"
        command = [sys.executable, str(script), str(config), "--output", str(output)]
        if ranks > 1:
            command = [launcher, "-n", str(ranks)] + command
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, cwd=tmp_path)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs[ranks] = output / "callbacks"
    serial, split = (pd.read_csv(outputs[r] / "losses.csv") for r in (1, 2))
    np.testing.assert_array_equal(serial["loss_value"].to_numpy(), split["loss_value"].to_numpy())
    for name in ("glifs_glifs_edges.h5", "virts_glifs_edges.h5"):
        serial_file, split_file = (outputs[r] / "trained_weights" / name for r in (1, 2))
        with h5py.File(serial_file) as a, h5py.File(split_file) as b:
            group = next(iter(a["edges"]))
            for key in ("source_node_id", "target_node_id", "edge_type_id"):
                np.testing.assert_array_equal(a["edges"][group][key][()], b["edges"][group][key][()])
            np.testing.assert_allclose(b["edges"][group]["0/syn_weight"][()], a["edges"][group]["0/syn_weight"][()],
                                       rtol=1e-4, atol=1e-6)
