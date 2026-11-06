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
    # A heavy neuron in the middle must not leave ranks empty.
    owner = plan_partition(10, 4, in_degree=[0, 0, 0, 1000, 0, 0, 0, 0, 0, 0])
    assert np.all(np.diff(owner) >= 0) and set(owner) == {0, 1, 2, 3}
    assert np.bincount(plan_partition(6, 4, balance="neurons")).max() == 2
    for seed in range(50):
        rng = np.random.default_rng(seed)
        n, size = int(rng.integers(2, 40)), int(rng.integers(1, 8))
        if size > n:
            continue
        degree = rng.integers(0, 5, n) * (rng.random(n) < 0.3) * 1000
        owner = plan_partition(n, size, in_degree=degree)
        assert np.all(np.diff(owner) >= 0) and set(owner) == set(range(size))


def test_grouped_plan_keeps_groups_whole():
    groups = np.repeat(["a", "b", "c", "d", "e"], [7, 3, 9, 2, 4])
    owner = plan_partition(len(groups), 3, in_degree=np.ones(len(groups)), groups=groups)
    for label in np.unique(groups):
        assert len(set(owner[groups == label])) == 1
    assert set(owner) == {0, 1, 2}


def _reference_refinement(owner, load, sizes, edges, size, max_load, max_neurons):
    """Unvectorized best-improvement search: every move then every swap, scanned group by group."""
    def cut(candidate):
        return int(edges[candidate[:, None] != candidate[None, :]].sum())

    def admissible(candidate):
        return (np.all(np.bincount(candidate, minlength=size) > 0)
                and np.bincount(candidate, weights=load, minlength=size).max() <= max_load
                and np.bincount(candidate, weights=sizes, minlength=size).max() <= max_neurons)

    owner = np.array(owner)
    score = cut(owner)
    while True:
        best, best_score = None, score
        for group in range(len(owner)):
            for rank in range(size):
                if rank != owner[group]:
                    candidate = owner.copy()
                    candidate[group] = rank
                    if admissible(candidate) and cut(candidate) < best_score:
                        best, best_score = candidate, cut(candidate)
            for other in range(group + 1, len(owner)):
                if owner[group] != owner[other]:
                    candidate = owner.copy()
                    candidate[group], candidate[other] = owner[other], owner[group]
                    if admissible(candidate) and cut(candidate) < best_score:
                        best, best_score = candidate, cut(candidate)
        if best is None:
            return owner
        owner, score = best, best_score


@pytest.mark.parametrize("seed", range(12))
def test_group_refinement_equals_reference_search(seed):
    from bmtk.simulator.dpointnet.neuron_partition import refine_group_placement

    rng = np.random.default_rng(seed)
    n, size = int(rng.integers(4, 13)), int(rng.integers(2, 5))
    size = min(size, n)
    edges = rng.integers(0, 6, (n, n)) * (rng.random((n, n)) < 0.6)
    load = rng.integers(1, 20, n).astype(float)
    sizes = rng.integers(1, 10, n).astype(float)
    start = np.arange(n) % size
    rng.shuffle(start)
    slack = rng.choice([1.0, 1.2, 2.0])
    max_load = max(slack * load.sum() / size, np.bincount(start, weights=load).max())
    max_neurons = max(slack * sizes.sum() / size, np.bincount(start, weights=sizes).max())
    expected = _reference_refinement(start, load, sizes, edges, size, max_load, max_neurons)
    owner = refine_group_placement(start, load, sizes, edges, size, max_load, max_neurons)
    np.testing.assert_array_equal(owner, expected)


def test_communication_placement_cuts_fewer_edges_within_balance():
    from bmtk.simulator.dpointnet.neuron_partition import cut_edges

    # Eight equal groups in two tightly connected clusters {0,1,2,3} and {4,5,6,7}: the
    # greedy placement (alternating ranks) mixes them, the refined one keeps each cluster on one rank.
    groups = np.repeat(np.arange(8), 10)
    cluster = np.arange(8) // 4
    edges = np.where(cluster[:, None] == cluster[None, :], 50, 1)
    np.fill_diagonal(edges, 0)
    degree = np.full(groups.size, 10)
    report = {}
    owner = plan_partition(groups.size, 2, in_degree=degree, groups=groups, group_edges=edges, report=report)
    greedy = plan_partition(groups.size, 2, in_degree=degree, groups=groups)
    group_owner = owner[::10]
    assert len(set(group_owner[cluster == 0])) == 1 and len(set(group_owner[cluster == 1])) == 1
    assert report["cut_edges"] == cut_edges(edges, group_owner) == 32
    assert report["greedy_cut_edges"] == cut_edges(edges, greedy[::10]) == 816
    assert np.bincount(owner).tolist() == [40, 40]
    np.testing.assert_array_equal(
        plan_partition(groups.size, 2, in_degree=degree, groups=groups, group_edges=edges), owner)
    with pytest.raises(ValueError, match="square matrix"):
        plan_partition(groups.size, 2, in_degree=degree, groups=groups, group_edges=edges[:4])


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
    codes = np.arange(recurrent.n_nodes) % 3
    expected = np.zeros((3, 3), dtype=np.int64)
    np.add.at(expected, (codes[edges["source"]], codes[edges["target"]]), 1)
    shares = [recurrent.partition_edge_statistics((i, 3), group_codes=codes, n_groups=3) for i in range(3)]
    np.testing.assert_array_equal(sum(share[0] for share in shares), degree)
    np.testing.assert_array_equal(sum(share[1] for share in shares), expected)
    assert recurrent.partition_edge_statistics()[1] is None
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
    version = subprocess.run([launcher, "--version"], capture_output=True, text=True)
    if any(name in version.stdout + version.stderr for name in ("Open MPI", "OpenRTE", "open-mpi")):
        # Hosted runners have fewer cores than ranks; Open MPI 4 prints "OpenRTE" here.
        command.insert(1, "--oversubscribe")
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "",
        # Also via the environment (Open MPI 4 and 5); MPICH ignores these.
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        "PRTE_MCA_rmaps_default_mapping_policy": ":oversubscribe",
    }
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"neuron partition probe passed on {ranks} ranks" in result.stdout


def _feedforward_network(directory):
    """Excitatory/inhibitory network without inhibitory-to-excitatory edges, 1-4 ms delays.

    Split by ``ei``, the excitatory rank receives no edges from the other rank.
    """
    from bmtk.builder import NetworkBuilder

    rng = np.random.default_rng(8)
    net = NetworkBuilder("glifs")
    for ei, count, pop, params in (("e", 60, "e4Nr5a1", "354833767_glif_lif_asc_config.json"),
                                   ("i", 30, "i4Vip", "501570114_glif_lif_asc_config.json")):
        net.add_nodes(N=count, ei=ei, pop_name=pop, model_type="point_neuron",
                      model_template="glif_psc_double_alpha", dynamics_params=params)
    for source, target, params, sign in (("e", "e", "glif_e2e.json", 1), ("e", "i", "glif_e2i.json", 1),
                                         ("i", "i", "glif_i2i.json", -1)):
        edges = net.add_edges(source={"ei": source}, target={"ei": target},
                              connection_rule=lambda s, t: int(s.node_id != t.node_id and rng.random() < 0.5),
                              dynamics_params=params, model_template="static_synapse")
        edges.add_properties(names=["syn_weight", "delay"],
                             rule=lambda *_, sign=sign: (sign * rng.uniform(2.0, 12.0), float(rng.integers(1, 5))),
                             dtypes=[float, float])
    net.build()
    net.save(str(directory))
    virts = NetworkBuilder("virts")
    virts.add_nodes(N=40, ei="e", model_type="virtual")
    for target, params in (("e", "virt_e2e.json"), ("i", "virt_e2i.json")):
        edges = virts.add_edges(target=net.nodes(ei=target), connection_rule=lambda *_: int(rng.random() < 0.5),
                                dynamics_params=params, model_template="static_synapse", delay=2.0)
        edges.add_properties(names="syn_weight", rule=lambda *_: rng.uniform(10.0, 40.0), dtypes=float)
    virts.build()
    virts.save(str(directory))


def _partition_config(tmp_path, variant):
    import json

    example = Path(__file__).resolve().parents[3] / "examples"
    config = json.loads((example / "dpointnet_mpi" / "config.partition.json").read_text())
    config["manifest"]["$ASSET_DIR"] = str(example / "dpointnet_all2all")
    if variant == "feedforward":
        network = tmp_path / "network"
        _feedforward_network(network)
        config["manifest"]["$NETWORK_DIR"] = str(network)
        config["rnn_cell_params"]["max_delay"] = 2.5  # clips the 3-4 ms edges, local and remote
        config["run"]["distribution"] = {"strategy": "neurons", "group_by": "ei"}
    path = tmp_path / f"config.{variant}.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.skipif(
    os.environ.get("BMTK_RUN_MPI_TESTS") != "1",
    reason="Real MPI subprocess tests require explicit BMTK_RUN_MPI_TESTS=1.",
)
@pytest.mark.parametrize("variant", ["example", "feedforward"])
def test_partitioned_training_matches_single_process_on_gpu(tmp_path, variant):
    """Training under mpiexec -n 2 equals one process: losses and exported SONATA weights.

    ``feedforward`` also covers a rank without remote sources, grouped planning and
    delays clipped by ``max_delay`` on both local and remote edges.
    """
    pytest.importorskip("mpi4py")
    import h5py
    import pandas as pd
    from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available

    if not fused_cuda_available():
        pytest.skip("Neuron-partitioned training uses the GPU direct recorded adjoint.")
    launcher = shutil.which("mpiexec")
    assert launcher, "MPI tests require mpiexec."
    config = _partition_config(tmp_path, variant)
    script = Path(__file__).resolve().parents[3] / "examples" / "dpointnet_mpi" / "run_training.py"
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


def test_dataset_rows_are_read_through_bounded_windows():
    from bmtk.simulator.dpointnet.network_adaptor import SONATANetwork

    class Dataset:
        dtype = np.float32

        def __init__(self, values):
            self.values, self.spans = values, []

        def __getitem__(self, item):
            self.spans.append(item.stop - item.start)
            return self.values[item]

    values = np.arange(1000, dtype=np.float32)
    rows = np.random.default_rng(4).permutation(1000)[:300]
    rows = np.r_[rows, rows[:7]]  # repeats
    dataset = Dataset(values)
    np.testing.assert_array_equal(SONATANetwork._read_dataset_rows(dataset, rows, 64), values[rows])
    assert max(dataset.spans) <= 64


def test_detected_mpi_launch_without_matching_communicator_fails(monkeypatch):
    from bmtk.simulator.dpointnet import io_tools
    from bmtk.utils.io.ioutils import bmtk_world_comm

    monkeypatch.setenv("PMI_SIZE", "3")
    monkeypatch.setattr(bmtk_world_comm, "_comm", None)
    monkeypatch.setattr(type(bmtk_world_comm), "comm", property(lambda self: None))
    with pytest.raises(RuntimeError, match="MPI launch of 3 processes"):
        io_tools.launched_world_communicator()
    monkeypatch.setenv("PMI_SIZE", "1")
    assert io_tools.launched_world_communicator() is None


def test_seeded_rate_ties_give_one_shared_permutation():
    import tensorflow as tf
    from bmtk.simulator.dpointnet.loss_functions.loss_utils import compute_spike_rate_distribution_loss

    rates = tf.constant(np.r_[np.zeros(40), np.full(20, 3.0), np.linspace(1, 9, 40)], tf.float32)
    target = tf.constant(np.sort(np.random.default_rng(1).uniform(0, 10, 100)), tf.float32)

    def gradient(seed):
        with tf.GradientTape() as tape:
            tape.watch(rates)
            loss = tf.reduce_sum(compute_spike_rate_distribution_loss(rates, target, seed=seed))
        return tf.convert_to_tensor(tape.gradient(loss, rates)).numpy()

    seed = tf.constant([7, 1], tf.int64)
    np.testing.assert_array_equal(gradient(seed), gradient(seed))
    assert not np.array_equal(gradient(seed), gradient(tf.constant([7, 2], tf.int64)))


def test_partitioned_recurrent_network_is_edge_free_metadata(example_networks, monkeypatch):
    from bmtk.simulator.dpointnet.rnn_model import RNN

    recurrent, _, reference, _ = example_networks
    rnn = RNN.__new__(RNN)
    rnn._built_recurrent_net = None
    rnn._recurrent_networks = {recurrent.name: recurrent}
    rnn.neuron_partition = object()
    monkeypatch.setattr(type(recurrent), "read_edges", lambda *a, **k: pytest.fail("edges were read"))
    monkeypatch.setattr(type(recurrent), "to_dict", lambda *a, **k: pytest.fail("edges were read"))
    metadata = rnn.recurrent_network
    assert "synapses" not in metadata and metadata["n_nodes"] == reference["n_nodes"]
    np.testing.assert_array_equal(metadata["node_type_ids"], reference["node_type_ids"])


@pytest.mark.parametrize("state,clip,error", [
    ("ZeroStateModule", {}, None),
    ("RandomizedStateModule", {}, "initial_state module zero_state"),
    ("ZeroStateModule", {"clipnorm": 1.0}, "clipnorm"),
    ("ZeroStateModule", {"global_clipnorm": 1.0}, "global_clipnorm"),
    ("ZeroStateModule", {"clipvalue": 1.0}, None),
])
def test_partition_setup_rejects_rank_local_state_and_norm_clipping(state, clip, error):
    from types import SimpleNamespace
    from bmtk.simulator.dpointnet.training import TrainingEngine

    engine = SimpleNamespace(
        init_state=type(state, (), {})(),
        optimizer=SimpleNamespace(**{"clipnorm": None, "global_clipnorm": None, "clipvalue": None, **clip}),
    )
    if error is None:
        TrainingEngine._validate_neuron_partition_setup(engine)
    else:
        with pytest.raises(ValueError, match=error):
            TrainingEngine._validate_neuron_partition_setup(engine)
