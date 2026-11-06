"""Neuron-partitioned execution: one network split by neurons across MPI processes.

Every process (rank) owns a set of recurrent neurons, their incoming edges and
the masters/optimizer state of those edges. Edges whose source neuron belongs
to another rank become the reserved named spike input ``PARTITION_INPUT`` of
the owner's cell; the direct recorded adjoint's spike-exchange hooks feed it
with the other ranks' spikes and return its credit to them
(:class:`MPISpikeExchange`). Population statistics are formed on a global view
(:func:`gather_neurons`, :func:`neuron_weighted_mean`) so losses run
unchanged and identically on every rank.

This module holds the transport-level pieces: the partition plan, route
construction, the MPI exchange and the differentiable collectives. All
collectives use host buffers through ``mpi4py``, so ranks may share a node,
a GPU or span nodes.
"""

import hashlib

import numpy as np
import tensorflow as tf

from .direct_recorded_adjoint import SpikeExchange
from .io_tools import mpi_launch_size  # noqa: F401  (re-exported)

PARTITION_INPUT = "__partition_recurrence__"


MAX_LOAD_RATIO = 1.05
MAX_NEURON_RATIO = 1.10
MAX_REFINED_GROUPS = 4096


def cut_edges(group_edges, group_owner):
    """Edges between groups on different ranks."""
    group_owner = np.asarray(group_owner)
    return int(np.asarray(group_edges)[group_owner[:, None] != group_owner[None, :]].sum())


def refine_group_placement(group_owner, load, sizes, group_edges, size, max_load, max_neurons,
                           max_rounds=1000):
    """Reduce the edges cut by a group placement with single moves and pairwise swaps.

    Each round applies the admissible move or swap that removes the most cut
    edges (ties: lowest group, then moves before swaps, then lowest rank or
    partner), keeping every rank non-empty, its load at most ``max_load`` and
    its neurons at most ``max_neurons``; it stops when no change removes an
    edge. Deterministic and exact (integer edge counts).
    """
    owner = np.array(group_owner, dtype=np.int64)
    load = np.asarray(load, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    edges = np.asarray(group_edges, dtype=np.int64)
    n = owner.size
    if edges.shape != (n, n) or load.shape != (n,) or sizes.shape != (n,):
        raise ValueError("Group placement inputs must describe the same groups.")
    if np.any(edges < 0):
        raise ValueError("Group edge counts must be non-negative.")
    # Symmetric traffic between distinct groups: an edge is cut in either direction.
    between = edges + edges.T
    np.fill_diagonal(between, 0)
    groups = np.arange(n)
    ranks = np.arange(size)
    for _ in range(int(max_rounds)):
        members = np.zeros((n, size), dtype=np.int64)
        members[groups, owner] = 1
        # connection[g, r]: edges between group g and the other groups on rank r.
        connection = between @ members
        own = connection[groups, owner]
        rank_load = np.bincount(owner, weights=load, minlength=size)
        rank_neurons = np.bincount(owner, weights=sizes, minlength=size)
        rank_groups = np.bincount(owner, minlength=size)
        # Change in cut edges of moving g -> r (negative: fewer cut edges).
        move_change = own[:, None] - connection
        move_ok = ((ranks[None, :] != owner[:, None])
                   & (rank_groups[owner] > 1)[:, None]
                   & (rank_load[None, :] + load[:, None] <= max_load)
                   & (rank_neurons[None, :] + sizes[:, None] <= max_neurons))
        move_change = np.where(move_ok, move_change, 0)
        # Swaps g <-> h (h > g, on another rank); their mutual edges stay cut.
        across = connection[:, owner]
        swap_change = own[:, None] + own[None, :] - across - across.T + 2 * between
        load_delta = load[None, :] - load[:, None]
        neuron_delta = sizes[None, :] - sizes[:, None]
        swap_ok = ((owner[:, None] != owner[None, :]) & (groups[None, :] > groups[:, None])
                   & (rank_load[owner][:, None] + load_delta <= max_load)
                   & (rank_load[owner][None, :] - load_delta <= max_load)
                   & (rank_neurons[owner][:, None] + neuron_delta <= max_neurons)
                   & (rank_neurons[owner][None, :] - neuron_delta <= max_neurons))
        swap_change = np.where(swap_ok, swap_change, 0)
        best = min(move_change.min(initial=0), swap_change.min(initial=0))
        if best >= 0:
            break
        # Scan order: for each group its moves (by rank), then its swaps (by partner).
        candidates = []
        g, r = np.nonzero(move_change == best)
        if g.size:
            candidates.append((g[0] * (size + n) + r[0], "move", g[0], r[0]))
        g, h = np.nonzero(swap_change == best)
        if g.size:
            candidates.append((g[0] * (size + n) + size + h[0], "swap", g[0], h[0]))
        _, kind, g, other = min(candidates)
        if kind == "move":
            owner[g] = other
        else:
            owner[g], owner[other] = owner[other], owner[g]
    return owner.astype(np.int32)


def plan_partition(n_nodes, size, in_degree=None, groups=None, balance="edges", group_edges=None,
                   max_load_ratio=MAX_LOAD_RATIO, max_neuron_ratio=MAX_NEURON_RATIO, report=None):
    """Owner rank of every global recurrent neuron (TF order), as int32.

    Without ``groups`` each rank receives one contiguous block of the TF order,
    balanced by ``in_degree`` (``balance="edges"``) or by neuron count
    (``"neurons"``); canonical edges are target-major, so each rank's edges are
    then a contiguous slice of the global canonical order. ``groups`` (one
    integer label per neuron) keeps every group on one rank; groups are placed
    largest first on the least-loaded rank. With ``group_edges`` (edges from
    group ``s`` to group ``t`` over the sorted unique labels) that placement is
    then refined by :func:`refine_group_placement` to cut fewer edges, within
    ``max_load_ratio`` / ``max_neuron_ratio`` of the mean rank load / neurons
    (or the greedy placement's maxima, if larger). ``report`` (a dict) receives
    the cut edges. The plan is deterministic and must be computed identically
    on every rank.
    """
    n_nodes, size = int(n_nodes), int(size)
    if size < 1 or n_nodes < size:
        raise ValueError(f"Cannot split {n_nodes} neurons across {size} ranks.")
    if balance not in ("edges", "neurons"):
        raise ValueError(f"Unknown partition balance {balance!r}")
    if balance == "edges":
        if in_degree is None:
            raise ValueError("Edge balancing requires per-neuron in-degrees.")
        load = np.asarray(in_degree, dtype=np.float64)
        if load.shape != (n_nodes,) or np.any(load < 0):
            raise ValueError("in_degree must hold one non-negative count per neuron.")
        # One unit per neuron keeps edge-free neurons from piling onto one rank.
        load = load + max(load.mean(), 1.0) * 1e-3
    else:
        load = np.ones(n_nodes, dtype=np.float64)
    owner = np.empty(n_nodes, dtype=np.int32)
    if groups is None:
        # Boundary k (neurons before rank k) is the number of neurons whose load midpoint
        # lies below k/size of the total, then made strictly increasing while reserving
        # one neuron for every later rank.
        cumulative = np.cumsum(load)
        midpoints = cumulative - load / 2
        targets = cumulative[-1] * np.arange(1, size) / size
        bounds = np.searchsorted(midpoints, targets, side="left")
        previous = 0
        for k in range(1, size):
            previous = int(np.clip(bounds[k - 1], previous + 1, n_nodes - (size - k)))
            bounds[k - 1] = previous
        for rank, (start, stop) in enumerate(zip(np.r_[0, bounds], np.r_[bounds, n_nodes])):
            owner[start:stop] = rank
        return owner
    groups = np.asarray(groups)
    if groups.shape != (n_nodes,):
        raise ValueError("groups must hold one label per neuron.")
    labels, inverse = np.unique(groups, return_inverse=True)
    if len(labels) < size:
        raise ValueError(f"{len(labels)} groups cannot occupy {size} ranks.")
    group_load = np.bincount(inverse, weights=load, minlength=len(labels))
    rank_load = np.zeros(size)
    group_owner = np.empty(len(labels), dtype=np.int32)
    for group in sorted(range(len(labels)), key=lambda g: (-group_load[g], g)):
        rank = int(np.argmin(rank_load))
        group_owner[group] = rank
        rank_load[rank] += group_load[group]
    if group_edges is not None:
        group_edges = np.asarray(group_edges)
        if group_edges.shape != (len(labels), len(labels)):
            raise ValueError("group_edges must be a square matrix over the sorted unique group labels.")
        group_sizes = np.bincount(inverse, minlength=len(labels)).astype(np.float64)
        greedy_load = np.bincount(group_owner, weights=group_load, minlength=size)
        greedy_neurons = np.bincount(group_owner, weights=group_sizes, minlength=size)
        max_load = max(max_load_ratio * group_load.sum() / size, greedy_load.max())
        max_neurons = max(max_neuron_ratio * n_nodes / size, greedy_neurons.max())
        greedy_cut = cut_edges(group_edges, group_owner)
        group_owner = refine_group_placement(
            group_owner, group_load, group_sizes, group_edges, size, max_load, max_neurons,
        )
        if report is not None:
            report.update(greedy_cut_edges=greedy_cut, cut_edges=cut_edges(group_edges, group_owner),
                          edges=int(group_edges.sum()))
    owner[:] = group_owner[inverse]
    if len(np.unique(owner)) != size:
        raise ValueError("Group placement left a rank without neurons.")
    return owner


class NeuronPartition:
    """A partition plan seen from one rank."""

    def __init__(self, owner, rank):
        self.owner = np.asarray(owner, dtype=np.int32)
        self.owner.setflags(write=False)
        self.size = int(self.owner.max()) + 1
        self.rank = int(rank)
        if not 0 <= self.rank < self.size:
            raise ValueError(f"Rank {rank} outside the partition of {self.size} ranks.")
        self.n_nodes = int(self.owner.size)
        self.nodes = [np.flatnonzero(self.owner == r) for r in range(self.size)]
        if any(len(nodes) == 0 for nodes in self.nodes):
            raise ValueError("Every rank must own at least one neuron.")
        self.local_nodes = self.nodes[self.rank]
        self.n_local = int(self.local_nodes.size)
        self.position = np.full(self.n_nodes, -1, dtype=np.int64)
        for nodes in self.nodes:
            self.position[nodes] = np.arange(nodes.size)
        self.digest = hashlib.sha256(self.owner.astype("<i4").tobytes()).hexdigest()

    def is_local(self, global_ids):
        return self.owner[np.asarray(global_ids)] == self.rank

    def describe(self):
        return {
            "ranks": self.size, "rank": self.rank, "neurons": self.n_nodes,
            "local_neurons": self.n_local, "plan_sha256": self.digest,
        }


def _mpi():
    from mpi4py import MPI

    return MPI


class MPISpikeExchange(SpikeExchange):
    """Spike/credit transport of :data:`PARTITION_INPUT` over host MPI.

    ``remote_sources`` are the global ids of the other ranks' neurons this
    rank's partition input projects from, ascending: column ``k`` of the input
    is neuron ``remote_sources[k]``. Routes are agreed collectively at
    construction; afterwards every forward step exchanges the fired
    ``(sample, neuron)`` pairs each peer needs and every reverse step the dense
    FP32 credit of each peer's needed neurons (silent ones included).
    """

    def __init__(self, comm, partition, remote_sources, batch_size):
        self.comm = comm
        self.partition = partition
        self.batch = int(batch_size)
        sources = np.asarray(remote_sources, dtype=np.int64)
        if sources.size and (np.any(np.diff(sources) <= 0) or np.any(partition.is_local(sources))):
            raise ValueError("Remote sources must be ascending neurons of other ranks.")
        self.width = int(sources.size)
        size = partition.size
        owners = partition.owner[sources]
        # What I request from each owner, as their local positions (ascending).
        requests = [partition.position[sources[owners == peer]] for peer in range(size)]
        self.receive_columns = [np.flatnonzero(owners == peer) for peer in range(size)]
        requested = comm.alltoall([request.astype(np.int64) for request in requests])
        # What each peer requests from me: my local neuron positions.
        self.send_positions = [np.asarray(value, dtype=np.int64) for value in requested]
        if any(np.any((value < 0) | (value >= partition.n_local)) for value in self.send_positions):
            raise ValueError("Peer requested a neuron this rank does not own.")
        # Local neuron -> (peer, index in that peer's request list), as CSR.
        peers = np.concatenate([np.full(v.size, peer, np.int64) for peer, v in enumerate(self.send_positions)])
        slots = np.concatenate([np.arange(v.size, dtype=np.int64) for v in self.send_positions])
        neurons = np.concatenate(self.send_positions)
        order = np.argsort(neurons, kind="stable")
        self._fan_peers, self._fan_slots = peers[order], slots[order]
        self._fan_splits = np.searchsorted(neurons[order], np.arange(partition.n_local + 1))
        self.send_sizes = np.array([v.size for v in self.send_positions], dtype=np.int64)
        self.receive_sizes = np.array([c.size for c in self.receive_columns], dtype=np.int64)
        self.tick = None
        self.forward_calls = self.reverse_calls = 0

    # -- host collectives ---------------------------------------------------------------

    def _alltoallv(self, chunks, dtype):
        MPI = _mpi()
        counts = np.array([chunk.size for chunk in chunks], dtype=np.int64)
        received_counts = np.empty_like(counts)
        self.comm.Alltoall(counts, received_counts)
        send = np.concatenate(chunks).astype(dtype, copy=False) if chunks else np.empty(0, dtype)
        received = np.empty(int(received_counts.sum()), dtype=dtype)
        offsets = np.r_[0, np.cumsum(counts)[:-1]]
        received_offsets = np.r_[0, np.cumsum(received_counts)[:-1]]
        mpi_type = MPI.INT64_T if np.dtype(dtype) == np.int64 else MPI.FLOAT
        self.comm.Alltoallv([send, counts, offsets, mpi_type],
                            [received, received_counts, received_offsets, mpi_type])
        return np.split(received, np.cumsum(received_counts)[:-1])

    def host_begin(self):
        self.tick = 0
        self.forward_calls = self.reverse_calls = 0
        return np.int32(0)

    def host_forward(self, time, fired):
        """``fired`` ``[k, 2]`` (sample, local neuron) -> flat ``sample * width + column`` received."""
        if int(time) != self.tick:
            raise RuntimeError(f"Spike exchange expected step {self.tick}, got {int(time)}")
        fired = np.asarray(fired, dtype=np.int64).reshape(-1, 2)
        samples, neurons = fired[:, 0], fired[:, 1]
        starts, stops = self._fan_splits[neurons], self._fan_splits[neurons + 1]
        fan = stops - starts
        entry = np.repeat(starts - np.r_[0, np.cumsum(fan)[:-1]], fan) + np.arange(fan.sum())
        entry_samples = np.repeat(samples, fan)
        peers, slots = self._fan_peers[entry], self._fan_slots[entry]
        chunks = []
        for peer in range(self.partition.size):
            selected = peers == peer
            # Header: the step, so a desynchronized peer fails loudly.
            chunks.append(np.r_[self.tick, entry_samples[selected] * self.send_sizes[peer] + slots[selected]])
        received = self._alltoallv(chunks, np.int64)
        flat = []
        for peer, values in enumerate(received):
            if values.size == 0 or values[0] != self.tick:
                raise RuntimeError(f"Spike exchange step mismatch with rank {peer}")
            values = values[1:]
            width = self.receive_sizes[peer]
            if values.size:
                sample, slot = values // width, values % width
                flat.append(sample * self.width + self.receive_columns[peer][slot])
        self.tick += 1
        self.forward_calls += 1
        return np.concatenate(flat).astype(np.int64) if flat else np.empty(0, np.int64)

    def host_reverse(self, credit):
        """Dense credit ``[batch, width]`` of my remote sources -> ``[batch, n_local]`` of my neurons."""
        credit = np.asarray(credit, dtype=np.float32).reshape(self.batch, self.width)
        if not np.all(np.isfinite(credit)):
            raise FloatingPointError("Non-finite partition credit")
        chunks = [np.ascontiguousarray(credit[:, columns]).reshape(-1) for columns in self.receive_columns]
        received = self._alltoallv(chunks, np.float32)
        result = np.zeros((self.batch, self.partition.n_local), dtype=np.float32)
        for positions, values in zip(self.send_positions, received):
            if positions.size:
                result[:, positions] += values.reshape(self.batch, positions.size)
        self.reverse_calls += 1
        return result

    # -- SpikeExchange hooks ------------------------------------------------------------

    def begin(self):
        return tf.numpy_function(self.host_begin, [], tf.int32, stateful=True)

    def forward(self, time, spikes):
        fired = tf.where(spikes > 0)
        received = tf.reshape(
            tf.numpy_function(self.host_forward, [time, fired], tf.int64, stateful=True), [-1])
        frame = tf.scatter_nd(received[:, None], tf.ones(tf.shape(received), spikes.dtype),
                              [self.batch * self.width])
        return tf.reshape(frame, [self.batch, self.width])

    def reverse(self, time, credit):
        local = tf.numpy_function(self.host_reverse, [credit], tf.float32, stateful=True)
        return tf.reshape(local, [self.batch, self.partition.n_local])


def _allgather_columns(comm, partition, local):
    """Host: ``[rows, n_local]`` on every rank -> ``[rows, n_nodes]`` in global neuron order."""
    MPI = _mpi()
    local = np.ascontiguousarray(local)
    rows = local.shape[0]
    counts = np.array([rows * nodes.size for nodes in partition.nodes], dtype=np.int64)
    received = np.empty(int(counts.sum()), dtype=local.dtype)
    mpi_type = MPI.FLOAT if local.dtype == np.float32 else MPI.DOUBLE
    comm.Allgatherv([local.reshape(-1), mpi_type],
                    [received, counts, np.r_[0, np.cumsum(counts)[:-1]], mpi_type])
    result = np.empty((rows, partition.n_nodes), dtype=local.dtype)
    offset = 0
    for nodes, count in zip(partition.nodes, counts):
        result[:, nodes] = received[offset:offset + count].reshape(rows, nodes.size)
        offset += count
    return result


def gather_neurons(comm, partition, local):
    """Global view ``[..., n_nodes]`` of a per-neuron tensor ``[..., n_local]``.

    Every rank receives the same values; a loss computed from them is
    replicated, so the gradient of the global loss with respect to this rank's
    neurons is the slice of the incoming gradient at its own neurons (no
    reduction is needed or applied).
    """
    local_nodes = tf.constant(partition.local_nodes, tf.int64)

    @tf.custom_gradient
    def gather(value):
        shape = tf.shape(value)
        rows = tf.reshape(tf.cast(value, tf.float32), [-1, partition.n_local])
        result = tf.numpy_function(
            lambda array: _allgather_columns(comm, partition, array), [rows], tf.float32, stateful=True)
        result = tf.cast(tf.reshape(result, tf.concat([shape[:-1], [partition.n_nodes]], axis=0)), value.dtype)

        def gradient(upstream):
            return tf.gather(upstream, local_nodes, axis=-1)

        return result, gradient

    return gather(local)


def neuron_weighted_mean(comm, partition, local_mean):
    """Global per-neuron mean from each rank's mean over its own neurons (any shape)."""
    share = partition.n_local / partition.n_nodes

    def allreduce(array):
        array = np.ascontiguousarray(array, dtype=np.float64) * share
        result = np.empty_like(array)
        comm.Allreduce(array, result, op=_mpi().SUM)
        return result.astype(np.float32)

    @tf.custom_gradient
    def reduce(value):
        result = tf.numpy_function(allreduce, [tf.cast(value, tf.float32)], tf.float32, stateful=True)
        result = tf.cast(tf.reshape(result, tf.shape(value)), value.dtype)

        def gradient(upstream):
            return upstream * tf.cast(share, upstream.dtype)

        return result, gradient

    return reduce(local_mean)


def allreduce_sum(comm, value):
    """Host-side sum of a NumPy array over all ranks (float64 accumulation)."""
    value = np.ascontiguousarray(value, dtype=np.float64)
    result = np.empty_like(value)
    comm.Allreduce(value, result, op=_mpi().SUM)
    return result


PARTITION_OPTIONS = ("strategy", "balance", "group_by", "placement", "weights_export")


def neuron_partition_requested(distribution, world_size, launched_for_samples):
    """Whether a run uses neuron partitioning.

    Explicit ``run.distribution.strategy == "neurons"`` always does; without a
    ``distribution`` entry a multi-process launch (plain ``mpirun``) does, as
    BioNet/PointNet split cells under ``mpirun``. ``strategy == "mpi"`` (and its
    launcher) keeps sample parallelism.
    """
    if isinstance(distribution, dict) and distribution.get("strategy") == "neurons":
        return True
    return distribution is None and world_size > 1 and not launched_for_samples


def select_rank_gpu(comm):
    """Make one GPU visible per rank (node-local rank modulo GPUs), before TF uses a device.

    Ranks beyond the node's GPU count share GPUs round-robin. Returns a report.
    """
    from mpi4py import MPI

    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank, local_size = node.rank, node.size
    node.Free()
    gpus = tf.config.list_physical_devices("GPU")
    report = {"local_rank": local_rank, "local_size": local_size, "gpus_on_node": len(gpus)}
    if gpus:
        selected = gpus[local_rank % len(gpus)]
        tf.config.set_visible_devices([selected], "GPU")
        report["gpu"] = selected.name
        report["shared_gpu"] = local_size > len(gpus)
    return report


def _abort_on_uncaught_exception(comm):
    """End every rank when one raises: its peers would otherwise wait in a collective forever."""
    import sys

    previous = sys.excepthook
    if getattr(previous, "_neuron_partition_abort", False):
        return

    def abort(kind, value, traceback):
        previous(kind, value, traceback)
        sys.stderr.flush()
        comm.Abort(1)

    abort._neuron_partition_abort = True
    sys.excepthook = abort


class PartitionContext:
    """Per-rank state of a neuron-partitioned run (plan, ownership, transport, statistics)."""

    def __init__(self, comm, options=None):
        options = dict(options or {})
        unknown = set(options) - set(PARTITION_OPTIONS)
        if unknown:
            raise ValueError(f"Unknown neuron-partition options: {sorted(unknown)}")
        self.comm = comm
        self.rank, self.size = comm.rank, comm.size
        _abort_on_uncaught_exception(comm)
        self.balance = options.get("balance", "edges")
        self.group_by = options.get("group_by")
        self.placement = options.get("placement", "communication")
        if self.placement not in ("communication", "greedy"):
            raise ValueError('placement must be "communication" or "greedy".')
        self.placement_report = None
        self.weights_export = options.get("weights_export", "gather")
        if self.weights_export not in ("gather", "per_rank"):
            raise ValueError('weights_export must be "gather" or "per_rank".')
        self.device = select_rank_gpu(comm)
        self.partition = None
        self.canonical_ids = {}
        self.input_rows = {}
        self.remote_sources = None
        self.n_canonical_edges = None
        self._exchange = None

    # -- planning and loading -------------------------------------------------------------

    def build_network_dicts(self, recurrent, input_networks, partition_input_options, dt=1.0, max_delay=None):
        """Plan the partition (once) and build this rank's recurrent and input dicts.

        The partition input is appended last, so the remaining inputs keep their
        replay columns. Its delays are clipped as GLIF3Cell clips recurrent delays
        (NEST): to ``[dt, limit]`` with ``limit`` the whole network's largest
        recurrent delay, capped by ``max_delay`` when positive, so remote and local
        edges keep the delays they have in the unsplit network.
        """
        if self.partition is None:
            groups = codes = None
            n_groups = 0
            if self.group_by is not None:
                nodes = recurrent.get_nodes_df()
                if self.group_by not in nodes.columns:
                    raise ValueError(f"Partition group_by column {self.group_by!r} is not a node attribute.")
                groups = nodes[self.group_by].to_numpy()
                labels, codes = np.unique(groups, return_inverse=True)
                n_groups = len(labels)
                refine = self.placement == "communication" and self.size > 1
                if refine and n_groups > MAX_REFINED_GROUPS:
                    raise ValueError(
                        f"placement 'communication' supports at most {MAX_REFINED_GROUPS} groups, "
                        f"group_by {self.group_by!r} has {n_groups}; use placement 'greedy'."
                    )
                if not refine:
                    codes = None
            # One scan (split across ranks) counts incoming edges and, for communication-aware
            # placement, the edges between every pair of groups.
            in_degree, group_edges = recurrent.partition_edge_statistics(
                (self.rank, self.size), group_codes=codes, n_groups=n_groups,
            )
            in_degree = np.rint(allreduce_sum(self.comm, in_degree)).astype(np.int64)
            if group_edges is not None:
                group_edges = np.rint(allreduce_sum(self.comm, group_edges)).astype(np.int64)
                self.placement_report = {}
            owner = plan_partition(recurrent.n_nodes, self.size, in_degree=in_degree, groups=groups,
                                   balance=self.balance, group_edges=group_edges,
                                   report=self.placement_report)
            digests = self.comm.allgather(hashlib.sha256(owner.tobytes()).hexdigest())
            if len(set(digests)) != 1:
                raise RuntimeError("Ranks computed different neuron partitions.")
            self.partition = NeuronPartition(owner, self.rank)
            self.target_offset = np.r_[0, np.cumsum(in_degree)][:-1]
            self.n_canonical_edges = int(in_degree.sum())
        network, partition_input, self.canonical_ids = recurrent.partition_recurrent_dicts(
            self.partition, self.target_offset, partition_input_options,
        )
        local_max = max(np.max(network["synapses"]["delays"], initial=dt),
                        np.max(partition_input["delays"], initial=dt))
        network_max = float(self.comm.allreduce(float(local_max), op=_mpi().MAX))
        limit = network_max if max_delay is None or max_delay <= 0 else min(network_max, max_delay)
        partition_input["delays"] = np.clip(partition_input["delays"], dt, limit)
        self.delay_limit_ms = limit
        inputs = {}
        for adaptor in input_networks:
            inputs[adaptor.name], self.input_rows[adaptor.name] = adaptor.partition_input_dict(self.partition)
        inputs[PARTITION_INPUT] = partition_input
        self.remote_sources = partition_input["remote_sources"]
        self.endpoints = recurrent._partition_endpoints
        return network, inputs

    def describe(self):
        report = {"device": self.device}
        if self.partition is not None:
            report.update(self.partition.describe())
            report["remote_sources"] = int(self.remote_sources.size)
            if self.placement_report:
                report["placement"] = dict(self.placement_report)
            report["local_edges"] = {name: int(ids.size) for name, ids in self.canonical_ids.items()}
        return report

    # -- runtime -------------------------------------------------------------------------

    def spike_exchanges(self, batch_size):
        if self._exchange is None or self._exchange.batch != int(batch_size):
            self._exchange = MPISpikeExchange(self.comm, self.partition, self.remote_sources, batch_size)
        return {PARTITION_INPUT: self._exchange}

    def global_counts(self, counts):
        return gather_neurons(self.comm, self.partition, counts)

    def global_mean(self, local_mean):
        return neuron_weighted_mean(self.comm, self.partition, local_mean)

    def gather_canonical(self, values_by_master, root=0):
        """Global canonical vector of per-edge values (``{master: local values}``) on ``root``."""
        ids = np.concatenate([self.canonical_ids[name] for name in values_by_master])
        values = np.concatenate([np.asarray(values_by_master[name]).reshape(-1) for name in values_by_master])
        gathered = self.comm.gather((ids, values), root=root)
        if self.rank != root:
            return None
        result = np.empty(self.n_canonical_edges, dtype=values.dtype)
        seen = np.zeros(self.n_canonical_edges, dtype=bool)
        for peer_ids, peer_values in gathered:
            result[peer_ids] = peer_values
            seen[peer_ids] = True
        if not seen.all():
            raise RuntimeError("Gathered canonical edges do not cover the network.")
        return result

    def broadcast_batch(self, values, root=0):
        """Rank ``root``'s input batch (nested tensors/arrays) on every rank."""
        def to_host(value):
            return value.numpy() if hasattr(value, "numpy") else value

        values = self.comm.bcast(
            tf.nest.map_structure(to_host, values) if self.rank == root else None, root=root,
        )
        return tf.nest.map_structure(
            lambda value: tf.convert_to_tensor(value) if isinstance(value, np.ndarray) else value, values,
        )


class PartitionedGroupedEMD:
    """Exact grouped Wasserstein-1 weight prior over masters split across ranks.

    Equals :class:`~bmtk.simulator.dpointnet.loss_functions.EMDWeightRegularization`
    of the unsplit network: the mean over connection groups of the mean absolute
    deviation between a group's sorted current and sorted initial weights, ties
    ordered by global canonical edge index. Each group is owned by the rank
    holding most of its edges; every step the other ranks send that group's
    values to its owner and receive their elements' derivatives back (groups
    entirely on their owner need no traffic).

    ``group_keys`` maps each master to the per-edge connection key of its edges
    in canonical order; ``masters`` maps each master to ``(variable, layout)``.
    """

    def __init__(self, context, masters, group_keys, dtype=tf.float32, local_groups="auto"):
        comm, partition = context.comm, context.partition
        self.comm, self.size, self.dtype = comm, partition.size, tf.as_dtype(dtype)
        names = list(masters)
        self.variables = [masters[name][0] for name in names]
        keys = {name: np.asarray(group_keys[name], dtype=np.int64) for name in names}
        unique = np.unique(np.concatenate(comm.allgather(np.unique(np.concatenate(list(keys.values()))))))
        self.n_groups = int(unique.size)
        group_dtype = np.uint16 if self.n_groups <= np.iinfo(np.uint16).max else np.int32
        groups = {name: np.searchsorted(unique, keys[name]).astype(group_dtype) for name in names}
        del keys
        local_counts = np.bincount(np.concatenate(list(groups.values())), minlength=self.n_groups)
        counts = np.stack(comm.allgather(local_counts))
        if local_groups not in ("auto", True, False):
            raise ValueError('local_groups must be "auto", True or False.')
        all_local = bool(np.all((counts > 0).sum(axis=0) == 1))
        if local_groups is True and not all_local:
            raise ValueError("local_groups=True requires every weight group on a single rank.")
        self._local = None
        if all_local and local_groups != False:
            self._local = _LocalGroupedEMD(context, masters, groups, self.n_groups, counts.sum(axis=0))
            self.report = {"groups": self.n_groups, "owned_groups": int(np.sum(local_counts > 0)),
                           "sent_elements": 0, "mode": "local"}
            return
        group_owner = np.argmax(counts, axis=0)
        group_size = counts.sum(axis=0)
        # Elements: (live row in the concatenated live masters, group, canonical id).
        live_rows, element_groups, element_ids, offset = [], [], [], 0
        for name, (variable, layout) in masters.items():
            n = int(variable.shape[0])
            canonical_to_live = (np.arange(n) if layout is None
                                 else np.asarray(layout.canonical_to_csr, dtype=np.int64))
            live_rows.append(canonical_to_live + offset)
            element_groups.append(groups[name])
            element_ids.append(np.asarray(context.canonical_ids[name], dtype=np.int64))
            offset += n
        live_rows = np.concatenate(live_rows)
        element_groups = np.concatenate(element_groups)
        element_ids = np.concatenate(element_ids)
        destination = group_owner[element_groups]
        send_order = np.lexsort((element_ids, element_groups, destination))
        self.send_rows = live_rows[send_order]
        self.send_counts = np.bincount(destination, minlength=self.size).astype(np.int64)
        self.n_live = offset
        # Owner side: who sends what (agreed once).
        received_ids = self._exchange(
            [element_ids[send_order][destination[send_order] == peer] for peer in range(self.size)], np.int64,
        )
        received_groups = self._exchange(
            [element_groups[send_order][destination[send_order] == peer] for peer in range(self.size)], np.int64,
        )
        self.receive_counts = np.array([chunk.size for chunk in received_ids], dtype=np.int64)
        ids, owned = np.concatenate(received_ids), np.concatenate(received_groups)
        self.arrange = np.lexsort((ids, owned))
        self.arranged_groups = owned[self.arrange]
        initial = np.concatenate([np.asarray(v.numpy(), np.float32).reshape(-1) for v in self.variables])
        sent, bounds = initial[self.send_rows], np.r_[0, np.cumsum(self.send_counts)]
        received_initial = np.concatenate(self._exchange(
            [sent[bounds[p]:bounds[p + 1]] for p in range(self.size)], np.float32,
        ))[self.arrange]
        order = np.lexsort((np.arange(received_initial.size), received_initial, self.arranged_groups))
        self.sorted_initial = received_initial[order]
        self.scale = (1.0 / (group_size[self.arranged_groups] * float(self.n_groups))).astype(np.float32)
        self.report = {"groups": self.n_groups, "owned_groups": int(np.sum(group_owner == partition.rank)),
                       "sent_elements": int(self.send_counts.sum() - self.send_counts[partition.rank]),
                       "mode": "exchange"}

    def _exchange(self, chunks, dtype):
        MPI = _mpi()
        counts = np.array([chunk.size for chunk in chunks], dtype=np.int64)
        received_counts = np.empty_like(counts)
        self.comm.Alltoall(counts, received_counts)
        send = np.concatenate(chunks).astype(dtype) if chunks else np.empty(0, dtype)
        received = np.empty(int(received_counts.sum()), dtype=dtype)
        mpi_type = MPI.INT64_T if np.dtype(dtype) == np.int64 else MPI.FLOAT
        self.comm.Alltoallv([send, counts, np.r_[0, np.cumsum(counts)[:-1]], mpi_type],
                            [received, received_counts, np.r_[0, np.cumsum(received_counts)[:-1]], mpi_type])
        return np.split(received, np.cumsum(received_counts)[:-1])

    def host_value_and_derivative(self, values):
        """Live master values (concatenated) -> (global loss, derivative per live element)."""
        values = np.asarray(values, dtype=np.float32)
        sent = values[self.send_rows]
        bounds = np.r_[0, np.cumsum(self.send_counts)]
        received = np.concatenate(self._exchange([sent[bounds[p]:bounds[p + 1]] for p in range(self.size)],
                                                 np.float32))
        arranged = received[self.arrange]
        order = np.lexsort((np.arange(arranged.size), arranged, self.arranged_groups))
        deviation = arranged[order] - self.sorted_initial
        local = np.array([np.sum(np.abs(deviation).astype(np.float64) * self.scale)], dtype=np.float64)
        total = np.empty_like(local)
        self.comm.Allreduce(local, total, op=_mpi().SUM)
        derivative_arranged = np.empty_like(arranged)
        derivative_arranged[order] = np.sign(deviation) * self.scale
        derivative_received = np.empty_like(received)
        derivative_received[self.arrange] = derivative_arranged
        rbounds = np.r_[0, np.cumsum(self.receive_counts)]
        returned = np.concatenate(self._exchange(
            [derivative_received[rbounds[p]:rbounds[p + 1]] for p in range(self.size)], np.float32,
        ))
        derivative = np.zeros(self.n_live, dtype=np.float32)
        derivative[self.send_rows] = returned
        return np.float32(total[0]), derivative

    def __call__(self):
        if self._local is not None:
            return tf.cast(self._local(), self.dtype)
        sizes = [int(v.shape[0]) for v in self.variables]

        @tf.custom_gradient
        def emd(*values):
            flat = tf.concat([tf.reshape(tf.cast(v, tf.float32), [-1]) for v in values], axis=0)
            loss, derivative = tf.numpy_function(
                self.host_value_and_derivative, [flat], (tf.float32, tf.float32), stateful=True,
            )
            loss = tf.reshape(loss, [])
            derivative = tf.reshape(derivative, [sum(sizes)])

            def gradient(upstream):
                parts = tf.split(derivative * tf.cast(upstream, tf.float32), sizes)
                return [tf.cast(part, v.dtype) for part, v in zip(parts, values)]

            return tf.cast(loss, self.dtype), gradient

        return emd(*[tf.convert_to_tensor(v) for v in self.variables])


class _LocalGroupedEMD:
    """:class:`PartitionedGroupedEMD` when every group lies on one rank, evaluated on device.

    Each rank sorts only its own groups, one group at a time inside the graph, so
    live memory is bounded by the largest group rather than the rank's edge count.
    Host setup keeps a compact (int32 where possible) group-ordered row list and the
    sorted initial values; derivatives go to per-master credit variables. The value
    and derivative equal the exchange route's: per-group mean absolute deviation of
    sorted current from sorted initial weights, ties by canonical edge index, divided
    by the global number of groups.
    """

    def __init__(self, context, masters, groups, n_groups, group_sizes):
        names = list(masters)
        self.comm = context.comm
        self.variables = [masters[name][0] for name in names]
        self.sizes = [int(v.shape[0]) for v in self.variables]
        n_live = sum(self.sizes)
        index_dtype = np.int32 if n_live < np.iinfo(np.int32).max else np.int64
        live_rows, element_groups, element_ids, offset = [], [], [], 0
        for name, size in zip(names, self.sizes):
            layout = masters[name][1]
            canonical_to_live = (np.arange(size, dtype=index_dtype) if layout is None
                                 else np.asarray(layout.canonical_to_csr).astype(index_dtype))
            live_rows.append(canonical_to_live + index_dtype(offset))
            element_groups.append(groups[name])
            element_ids.append(np.asarray(context.canonical_ids[name], dtype=np.int64))
            offset += size
        live_rows = np.concatenate(live_rows)
        element_groups = np.concatenate(element_groups)
        by_id = np.argsort(np.concatenate(element_ids), kind="stable")
        del element_ids
        order = by_id[np.argsort(element_groups[by_id], kind="stable")]
        del by_id
        self.order = live_rows[order]
        local_groups = element_groups[order]
        del order, live_rows, element_groups
        owned = np.flatnonzero(np.bincount(local_groups, minlength=n_groups))
        bounds = np.searchsorted(local_groups, owned, side="left")
        ends = np.searchsorted(local_groups, owned, side="right")
        del local_groups
        initial = np.concatenate([np.asarray(v.numpy(), np.float32).reshape(-1) for v in self.variables])
        self.sorted_initial = np.empty(len(self.order), dtype=np.float32)
        for start, stop in zip(bounds, ends):
            self.sorted_initial[start:stop] = np.sort(initial[self.order[start:stop]])
        del initial
        self.scales = (1.0 / (np.asarray(group_sizes, np.float64)[owned] * n_groups)).astype(np.float32)
        self.starts, self.stops = bounds.astype(np.int64), ends.astype(np.int64)
        with tf.device("/CPU:0"):
            self._order = tf.constant(self.order)
            self._initial = tf.constant(self.sorted_initial)
            self._starts = tf.constant(self.starts)
            self._stops = tf.constant(self.stops)
            self._scales = tf.constant(self.scales)
        with tf.init_scope():
            self.credits = [tf.Variable(tf.zeros(v.shape, tf.float32), trainable=False,
                                        name="emd_prior_credit") for v in self.variables]

    def _allreduce(self, value):
        return np.float32(self.comm.allreduce(float(value)))

    def __call__(self):
        sizes, credits = self.sizes, self.credits
        boundaries = np.cumsum([0] + sizes)
        n_owned = len(self.scales)

        @tf.custom_gradient
        def emd(*values):
            values = [tf.reshape(tf.cast(v, tf.float32), [-1]) for v in values]
            resets = [credit.assign(tf.zeros_like(credit)) for credit in credits]

            def body(i, total):
                with tf.device("/CPU:0"):
                    start, stop = self._starts[i], self._stops[i]
                    positions = self._order[start:stop]
                    initial = self._initial[start:stop]
                    scale = self._scales[i]
                positions = tf.cast(positions, tf.int64)
                current = None
                masks = []
                for k, value in enumerate(values):
                    low, high = int(boundaries[k]), int(boundaries[k + 1])
                    if high == low:
                        continue
                    inside = (positions >= low) & (positions < high)
                    local = tf.clip_by_value(positions - low, 0, high - low - 1)
                    gathered = tf.gather(value, local)
                    current = gathered if current is None else tf.where(inside, gathered, current)
                    masks.append((k, inside, local))
                order = tf.math.top_k(-current, k=tf.size(current), sorted=True).indices
                deviation = tf.gather(current, order) - initial
                derivative = tf.sign(deviation) * scale
                for k, inside, local in masks:
                    chosen = tf.gather(inside, order)
                    tf.raw_ops.ResourceScatterUpdate(
                        resource=credits[k].handle,
                        indices=tf.boolean_mask(tf.gather(local, order), chosen),
                        updates=tf.boolean_mask(derivative, chosen),
                    )
                return i + 1, total + tf.reduce_sum(tf.abs(deviation)) * scale

            with tf.control_dependencies(resets):
                _, local_total = tf.while_loop(
                    lambda i, _: i < n_owned, body, (tf.constant(0), tf.constant(0.0, tf.float32)),
                    parallel_iterations=1,
                )
            total = tf.numpy_function(self._allreduce, [local_total], tf.float32, stateful=True)
            total = tf.reshape(total, [])

            def gradient(upstream):
                with tf.control_dependencies([total]):
                    return [tf.cast(credit.read_value() * tf.cast(upstream, tf.float32), v.dtype)
                            for credit, v in zip(credits, values)]

            return total, gradient

        return emd(*[tf.convert_to_tensor(v) for v in self.variables])
