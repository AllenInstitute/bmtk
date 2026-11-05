"""Multi-rank oracle of the neuron-partition transport and collectives.

Run as ``mpiexec -n 3 python neuron_partition_probe.py``. Every check compares
with a dense reference formed from all ranks' data, so any rank count works.
Exits nonzero on the first mismatch.
"""

import sys

import numpy as np
import tensorflow as tf
from mpi4py import MPI

from bmtk.simulator.dpointnet.neuron_partition import (
    MPISpikeExchange, NeuronPartition, gather_neurons, neuron_weighted_mean, plan_partition,
)


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    tf.config.set_visible_devices([], "GPU")
    n_nodes, batch, steps = 41, 2, 4
    rng = np.random.default_rng(7)  # identical on every rank
    in_degree = rng.integers(0, 30, n_nodes)
    owner = plan_partition(n_nodes, size, in_degree=in_degree)
    assert np.array_equal(comm.bcast(owner, root=0), owner), "plan differs across ranks"
    partition = NeuronPartition(owner, rank)
    assert sorted(np.concatenate(partition.nodes).tolist()) == list(range(n_nodes))

    # Each rank needs a random subset of the other ranks' neurons.
    others = np.flatnonzero(owner != rank)
    local_rng = np.random.default_rng(100 + rank)
    sources = np.sort(local_rng.choice(others, size=min(len(others), 9), replace=False))
    exchange = MPISpikeExchange(comm, partition, sources, batch)
    all_sources = comm.allgather(sources)

    exchange.host_begin()
    for step in range(steps):
        spikes = (np.random.default_rng(1000 * step + rank).random((batch, partition.n_local)) < 0.4)
        everyone = np.zeros((batch, n_nodes), bool)
        for peer, peer_spikes in enumerate(comm.allgather(spikes)):
            everyone[:, partition.nodes[peer]] = peer_spikes
        fired = np.argwhere(spikes)
        flat = exchange.host_forward(step, fired)
        frame = np.zeros(batch * len(sources), bool)
        frame[flat] = True
        assert np.array_equal(frame.reshape(batch, -1), everyone[:, sources]), f"forward step {step}"

    credit = np.random.default_rng(50 + rank).standard_normal((batch, len(sources))).astype(np.float32)
    local = exchange.host_reverse(credit)
    expected = np.zeros((batch, n_nodes), np.float64)
    for peer_sources, peer_credit in zip(all_sources, comm.allgather(credit)):
        expected[:, peer_sources] += peer_credit
    np.testing.assert_allclose(local, expected[:, partition.local_nodes], rtol=1e-6, atol=1e-6)

    # TF hook path: one forward frame and one reverse credit.
    exchange.begin()
    spikes = (np.random.default_rng(77 + rank).random((batch, partition.n_local)) < 0.5).astype(np.float32)
    everyone = np.zeros((batch, n_nodes), np.float32)
    for peer, peer_spikes in enumerate(comm.allgather(spikes)):
        everyone[:, partition.nodes[peer]] = peer_spikes
    frame = exchange.forward(tf.constant(0), tf.constant(spikes)).numpy()
    np.testing.assert_array_equal(frame, everyone[:, sources])
    np.testing.assert_allclose(exchange.reverse(tf.constant(0), tf.constant(credit)).numpy(), local, rtol=1e-6)

    # Global view and its gradient: a loss replicated on every rank.
    counts = tf.constant(np.random.default_rng(9 + rank).random((batch, partition.n_local)), tf.float32)
    weights = tf.constant(np.random.default_rng(3).random((batch, n_nodes)), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(counts)
        view = gather_neurons(comm, partition, counts)
        loss = tf.reduce_sum(tf.square(view) * weights)
    gradient = tape.gradient(loss, counts).numpy()
    reference = np.zeros((batch, n_nodes), np.float32)
    for peer, peer_counts in enumerate(comm.allgather(counts.numpy())):
        reference[:, partition.nodes[peer]] = peer_counts
    np.testing.assert_array_equal(view.numpy(), reference)
    np.testing.assert_allclose(gradient, (2 * reference * weights.numpy())[:, partition.local_nodes], rtol=1e-6)
    assert np.allclose(comm.allgather(float(loss)), float(loss)), "replicated loss differs"

    local_mean = tf.constant(np.random.default_rng(5 + rank).random((batch, 3)), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(local_mean)
        mean = neuron_weighted_mean(comm, partition, local_mean)
        total = tf.reduce_sum(mean)
    shares = np.array([len(nodes) / n_nodes for nodes in partition.nodes])
    expected_mean = sum(s * m for s, m in zip(shares, comm.allgather(local_mean.numpy())))
    np.testing.assert_allclose(mean.numpy(), expected_mean, rtol=1e-6)
    np.testing.assert_allclose(tape.gradient(total, local_mean).numpy(), shares[rank], rtol=1e-6)
    check_grouped_emd(comm, partition, owner)
    check_grouped_emd(comm, partition, owner, rank_local_groups=True)
    comm.Barrier()
    if rank == 0:
        print(f"neuron partition probe passed on {size} ranks")


def check_grouped_emd(comm, partition, owner, rank_local_groups=False):
    """The rank-split grouped EMD equals the single-process grouped EMD (value and gradient).

    With ``rank_local_groups`` every group's edges share a target owner, so the device
    evaluation of rank-local groups is used instead of the exchange route.
    """
    from types import SimpleNamespace

    from bmtk.simulator.dpointnet.loss_functions.weight_regularization import (
        _build_group_order, _grouped_emd_custom_gradient, _sort_initial_values_by_group,
    )
    from bmtk.simulator.dpointnet.neuron_partition import PartitionedGroupedEMD

    rng = np.random.default_rng(21)  # identical global network on every rank
    n_nodes, edges = partition.n_nodes, 300
    target, source = rng.integers(0, n_nodes, edges), rng.integers(0, n_nodes, edges)
    order = np.lexsort((np.arange(edges), source, target))
    target, source = target[order], source[order]  # canonical order
    pop = rng.integers(0, 3, n_nodes)
    keys = pop[target] * 1000 + pop[source]
    if rank_local_groups:
        keys = keys + owner[target].astype(np.int64) * 1_000_000
    initial = np.round(rng.standard_normal(edges), 1).astype(np.float32)  # ties
    current = np.round(initial + rng.standard_normal(edges) * 0.3, 1).astype(np.float32)

    unique, group_ids = np.unique(keys, return_inverse=True)
    group_order, splits = _build_group_order(group_ids, len(unique))
    sorted_initial = _sort_initial_values_by_group(initial, group_order, splits)
    slices = tuple((int(splits[i]), int(splits[i + 1])) for i in range(len(unique)))
    denominators = tuple(int((e - s) * len(unique)) for s, e in slices)
    x = tf.Variable(current)
    with tf.GradientTape() as tape:
        reference = _grouped_emd_custom_gradient(
            tf.convert_to_tensor(x), tf.constant(group_order), tf.constant(sorted_initial), slices, denominators)
    reference_gradient = tape.gradient(reference, x).numpy()

    mine = np.flatnonzero(owner[target] == partition.rank)
    local_source = owner[source[mine]] == partition.rank
    masters, group_keys, canonical_ids, endpoints = {}, {}, {}, {}
    variables = []
    for name, selected in (("<recurrent>", mine[local_source]), ("__partition_recurrence__", mine[~local_source])):
        live = np.random.default_rng(31 + partition.rank).permutation(selected.size)  # live row of canonical i
        layout = SimpleNamespace(canonical_to_csr=live)
        values = np.empty(selected.size, np.float32)
        values[live] = initial[selected]
        variable = tf.Variable(values)
        masters[name] = (variable, layout)
        variables.append((variable, live, selected))
        group_keys[name] = keys[selected]
        canonical_ids[name] = selected
        endpoints[name] = (target[selected], source[selected])
    context = SimpleNamespace(comm=comm, partition=partition, canonical_ids=canonical_ids, endpoints=endpoints)
    emd = PartitionedGroupedEMD(context, masters, group_keys)
    expected_mode = "local" if rank_local_groups or partition.size == 1 else "exchange"
    assert emd.report["mode"] == expected_mode, emd.report
    for variable, live, selected in variables:
        values = np.empty(selected.size, np.float32)
        values[live] = current[selected]
        variable.assign(values)
    with tf.GradientTape() as tape:
        value = emd()
    gradients = tape.gradient(value, [v for v, _, _ in variables])
    np.testing.assert_allclose(float(value), float(reference), rtol=1e-6)
    for gradient, (_, live, selected) in zip(gradients, variables):
        np.testing.assert_allclose(gradient.numpy()[live], reference_gradient[selected], rtol=1e-6, atol=1e-9)
    assert np.allclose(comm.allgather(float(value)), float(value)), "EMD value differs across ranks"

    @tf.function
    def compiled():
        with tf.GradientTape() as tape:
            result = emd()
        return result, tape.gradient(result, [v for v, _, _ in variables])

    value, gradients = compiled()
    np.testing.assert_allclose(float(value), float(reference), rtol=1e-6)
    for gradient, (_, live, selected) in zip(gradients, variables):
        np.testing.assert_allclose(gradient.numpy()[live], reference_gradient[selected], rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        MPI.COMM_WORLD.Abort(1)
