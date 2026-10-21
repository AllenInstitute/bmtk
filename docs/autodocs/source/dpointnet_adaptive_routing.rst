Adaptive routing for the CSR-master sweep
=========================================

``rnn_cell_params.use_adaptive_routing`` selects the ``adaptive_segment_ticket_v1``
forward routing schema for the combined current sweep with CSR-ordered FP32
masters and uint8 synapse types (:doc:`dpointnet_csr_masters`). It is selected
automatically with uint8 types at batch 8 or larger
(``acceleration.ADAPTIVE_ROUTING_MIN_BATCH``, measured slower at batch 1) when the
loaded library advertises the capability; see :doc:`dpointnet_performance`. It is an independent implementation of a
compressed-segment, registry-shared, adaptively scheduled projection design.

Selection and errors
--------------------

``false`` keeps the dense tile sweep on the same binaries. Explicit ``true``
without uint8 types, native fused CUDA or the combined current sweep raises
``ValueError`` during construction. A native library without the
``adaptive_routing``/``adaptive_traversal`` attributes of
``dpointnet_combined_sweep_csr_master`` or without ``dpointnet_routing_census``
raises ``RuntimeError`` for explicit ``true`` and is skipped by automatic
selection. The low-level ``fused_combined_currents(..., adaptive_routing=True)``
requires a boolean, ``csr_master_weights=True`` and uint8 type metadata.

The acceleration report records ``adaptive_routing`` (``"adaptive_segment_ticket_v1"``
or ``"disabled"``) and, when enabled, ``adaptive_routing_plan`` (schema,
wide/narrow tile and per-source traversal); ``cell.adaptive_routing_plan()``
returns the same static plan. ``routing_geometry`` (and
``cell.combined_routing_geometry()``) gives each source's static traversal and
tile: ``dense_tile_starts`` with one ``tile`` when disabled, or the adaptive
traversal with ``wide_tile``/``narrow_tile`` and the per-launch rule when enabled.
With adaptive routing the automatic wide tile is 2048 at every batch, so the
quarter-width narrow tile (512) is available at small batches. The per-launch rule
is mirrored by ``csr_spike_ops.adaptive_selected_tile(batch, n_post, wide_tile,
multiprocessors, wide_blocks_per_sm)``; ``combined_routing_census`` reports the
tile actually selected (``last_selected_tile``) and built (``built_tiles``), which
differ between devices (for example, batch 32 at 4,099 targets stays wide on an
RTX 3090 but is narrow on an A100 or L40S).

Operator and kernel names are unchanged for profilers: the forward op remains
``DpointnetCombinedSweepCsrMaster`` and the adaptive kernel is
``tensorflow::adaptive_routing::DpointnetCombinedSweepKernel<T, Index, tile, threads>``
(it contains ``DpointnetCombinedSweepKernel``). Its queue kernels (cub
``DeviceSelectSweepKernel``/``DeviceCompactInitKernel`` and
``AdaptiveSampleStartsKernel``) run under the same op. Per-source VJP ops and
kernels are unchanged. The only new op is the host-output
``DpointnetRoutingCensus`` used by the census API.

What changes
------------

Only the forward joint sweep changes. Each source's canonical VJP, live
silent-neuron spike credit, weight gradients, carrier seeds, delays, Poisson
counts, constraints, checkpoints/export and master ownership are unchanged.
The forward arithmetic is unchanged per edge: ``fmul_rn(spike, weight)`` times
each basis coefficient, accumulated in FP32 shared target tiles and rounded once
to the compute dtype. Only the order of FP32 atomic additions differs, so
results are not bitwise equal to the disabled schema but agree with independent
FP64 oracles at the existing tolerances.

The backward keeps its existing routing deliberately: in the reference design
the tile segments, active-row queues and tickets are forward-only as well; its
spike VJP uses a basis-pair projection and its weight gradient an event-row
queue, and both kinds of consumers already exist independently in this
source's per-source VJPs.

Representation (native-owned)
-----------------------------

For each (metadata allocation, tile width) the operator builds, on first use and
on the device, a compressed table of each row's nonempty tile segments:
``splits[width + 1]`` (segment index range per row) and ``starts[S + 1]`` (first
CSR position per segment, with ``starts[S] = n_edges`` so a segment ends where
the next begins). Up to 64 tiles, one 64-bit mask per row names the row's tiles
and the segment rank is a popcount, so no per-segment tile ids are stored. Above
64 tiles a ``tiles[S]`` array is stored and each row's tiles are binary-searched.
Target-major gather tables for gathered sources use the existing construction.
Table construction synchronizes the stream once to read the segment total;
warm launches perform no host transfers.

Tables live in a process registry of weak references keyed by device, the actual
metadata allocation, element count, source width, edge count, target count,
type schema and tile width. Each operator holds strong references to at most two
segment tables per source (wide and narrow) and one gather table per gathered
source; entries also hold the metadata tensor so the address cannot be reused.
Operators sharing a connectivity (for example forward and recompute graphs)
share tables. A table is released when the last operator holding it is
destroyed; TensorFlow decides when function kernels are destroyed (deleting the
functions and clearing the eager kernel cache releases them). Nothing is cached
per batch: active-row queues are per-launch temporaries.

Changing a source to a different connectivity resource refreshes that source's
tables. Assigning new values to the same metadata resource is rejected with
``InvalidArgumentError`` ("immutable connectivity"); rebuild the connectivity
instead. Live weight updates are read on every launch.

Adaptive policy
---------------

Per source, fixed at graph construction by ``adaptive_routing_plan``:

* ``gather``: gather requested and the source is gather-eligible (mean in-degree
  at most 16), as for the existing target-major gather;
* ``warp_batches``: rows never repeat a target and mean fanout is below 512;
* ``block_rounds``: all other sources (repeated or unknown target repetition,
  or high fanout).

These supersede the cooperative-row and fanout-round requests inside the
adaptive schema; both remain in effect with ``use_adaptive_routing=false``.

Per launch, on the host from the actual batch: the configured
``combined_current_target_tile_size`` is the wide tile. A narrow tile (a quarter
of the wide one, clamped to 256..1024; 2048 maps to 512) is used when
``2 * batch * wide_tiles`` is at most the device's resident block slots for the
wide kernel (multiprocessors times its occupancy); otherwise the wide tile.

Per block, on the device: active rows come from an ordered compaction of each
non-gathered source's ``[batch, width]`` activity with per-sample starts.
``warp_batches`` warps claim batches of 4 to 32 consecutive active rows from a
per-block atomic ticket (about eight batches per warp from the sample's active
count), compact the rows with a segment in the tile and walk their concatenated
edges with two independent edges per lane per step. ``block_rounds`` compacts a
round of found rows with one packed block scan and sums equal-target runs in
registers before the shared atomics. Gathered sources use the existing
one-owner-per-target accumulation.

Routing census
--------------

``cell.adaptive_routing_report()`` returns, per source name (``recurrent`` then
named inputs), the mode, policy and an exact census read from native bookkeeping
without reading or copying routing data:
``dense_tile_bytes``/``dense_tile_tables`` (disabled schema, per operator),
``segment_bytes``, ``mask_bytes``, ``segment_tables``, ``segments``,
``built_tiles``, ``gather_table_bytes``/``gather_tables``,
``transient_queue_bytes`` (last launch, summed over operators),
``last_selected_tile``, ``traversal`` and ``routing_table_bytes`` (sum of the
persistent table bytes). Totals cover every live operator instance keyed by the
source's current metadata allocation and are zero before the first execution.
``csr_spike_ops.combined_routing_census(connectivities)`` exposes the same
fields for low-level use. Byte counts are tensor payloads, not allocator
capacity or driver reservations.

Limitations
-----------

* Requires ``batch * source_width < 2**31`` per non-gathered source; larger
  launches raise ``InvalidArgumentError``.
* Carried per-delay-slot queue records are not used; each launch compacts the
  full ``[batch, width]`` activity of each non-gathered source.
* Background/neuron-update fusion is not part of this schema.
* Kernel execution has been validated on RTX 3090 (SM86). All seven compiled
  targets are admitted automatically; their launch geometry is checked from the
  compute capability (512-thread blocks with static shared memory below 48 KiB for
  automatic tiles, occupancy measured at launch), but execution on other GPU
  families has not been separately qualified.
