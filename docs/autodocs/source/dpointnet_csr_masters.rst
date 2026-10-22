CSR-ordered FP32 master weights: implementation reference
=========================================================

``rnn_cell_params.use_csr_master_weights`` stores every recurrent and named-input
FP32 master in stable CSR (source-major) edge order and lets the native kernels
read it directly. It is selected automatically inside an admitted joint current
sweep; :doc:`dpointnet_performance` describes the selection policy and
:doc:`dpointnet_training_standards` the defaults and migration. This page is the
implementation reference for the layout, metadata, ownership and canonical
boundaries. ``use_uint8_synapse_types`` and ``use_adaptive_routing`` build on it.

Explicit ``true`` requires NEST dynamics, compute temporal gradients, per-edge
recurrent weights, one replica, FP32 masters, spike or count inputs and, with
native CUDA, the combined current sweep; otherwise construction raises. Without
native CUDA and with no visible GPU, explicit ``true`` selects a CPU reference
implementation used for correctness tests; construction raises when the loaded
operator library lacks its ``dpointnet_master_projection`` reference ops (for
example a library built before this option existed). These reference ops reject
metadata whose splits, post indices, synapse types or canonical edge IDs fall
outside the operand dimensions; their GPU registration (tests only) validates
with one flag read-back per call and is never launched by the combined sweep.
``false`` keeps canonical-order masters with their compute shadows. The acceleration report records
``csr_master_weights`` (``direct_csr_fp32_master_host_lifecycle_v1`` or
``disabled``) and ``csr_master_projection``
(``optimized_combined_csr_fp32_master``, ``cpu_reference_csr_fp32_master`` or
``disabled``).

Host-prepared lifecycle
-----------------------

The lifecycle keeps exactly one read-only NumPy ``csr_to_canonical`` mapping
(uint32 for admitted edge counts; uint64 beyond the uint32 range).
``canonical_to_csr`` is a fresh NumPy inverse on demand, never cached. The
connectivity and layout share that same mapping array. There is no eager
TensorFlow permutation constant for TensorFlow to mirror on the GPU.
Initialization reorders weights, constraint signs, delay-expanded endpoints,
synapse types and export factors on the host. Endpoint variables and type
constants remain CPU-resident in live CSR order; necessary native routing stays
device-local. Prepared endpoint variables are assigned that runtime order.

Native resources omit the unused canonical edge-ID segment only for lifecycle
connectivity. Layout is ``posts, types, splits, [pair_ids, pair_posts,
pair_types]``; compact consumers recognize the exact shorter shape, never infer
an identity edge-ID table. All direct forward, spike/carrier VJPs, weight-only
VJPs, queues and target-gather routing support it. Canonical/reorder/restore
consumers reject the shorter shape. The unused fixed-four input-forward tables
are not built for this VJP-only producer path. CSC edge positions used by the
actual target-gather forward are not canonical edge IDs and are retained. Both fat operators
must be rebuilt; an old library rejects the compact shape.

Canonical hashing/export/assignment, reference preparation, index composition
and explicit checkpoint migration use NumPy host boundaries. Canonical export
and index composition reject graph tracing. ``weight_loss_values`` still returns
the exact differentiable CSR master; compose EMD indices only once before
tracing, as below. Low-level ``layout.to_canonical`` remains an explicit
differentiable permutation for CPU native reference probes and tests; it is not
the host export API and must not be inserted into a timed native loss.

Checkpoints never rewrite live state: every CSR-ordered checkpointed variable
(masters, endpoint indices and paired optimizer slots) serializes a host-converted
canonical copy, as described under the checkpoint section below.
``cell.csr_host_lifecycle_metadata()`` reports exact owned payload bytes per
surface for mapping, native resource, incoming arrays, endpoints, types and
export factors. These are not allocator peaks, cached mirrors or driver memory.
For native cells the report also includes ``metadata_shape``, ``metadata_schema``,
``canonical_edge_ids_present``, ``canonical_edge_id_offset`` and existing resource
object IDs/devices for a noncopying census. The corresponding connectivity exposes
these static resource-creation receipts and ``metadata_resource_handles``. These
describe the actual packed resource allocation, not an empty wrapper argument.
Count a handle only once: master/fused connectivity attributes alias the same
resource. Reading the resource for independent shape validation is optional and
untimed; the census API itself never reads or copies its payload.

Supported execution is single-replica NEST with independently parameterized FP32
edge weights, spike/count inputs, compute temporal gradients and the optimized
native combined sweep. The CPU implementation is a correctness reference only.
FP32-temporal replay, distributed strategies, local learning rules, continuous
current inputs and alternate saved projection values use canonical-order masters.

Native uint8 synapse types
--------------------------

``use_uint8_synapse_types`` stores per-edge synapse types as a byte tail of the
same metadata resource. It requires CSR masters and the native combined route;
automatic selection additionally requires every recurrent and input type ID to be
within 0..255 and all five native type consumers to accept the byte tail. ``false``
keeps wide per-edge types. The report marker ``csr_uint8_types`` is
``csr_live_uint8_type_tail_v1`` or ``disabled``.

Every per-edge ID must have an integer dtype and be nonnegative, no larger
than 255, and inside the declared basis table before any uint8 conversion.
Fractional, nonfinite, Boolean, complex, string and out-of-range IDs reject
before resource allocation. Valid larger IDs use the wide representation in automatic
selection; explicit typed mode never truncates them.
Type-table rows and pair-type IDs keep their existing interpretation.

There is still one packed resource per surface with the original index dtype
(``uint32`` or ``int64``). Its mixed-width schema is index-width
``posts[E], splits[S+1], [pair_ids[E], pair_posts[P], pair_types[P]]``,
then exactly ``E`` uint8 type bytes and zero tail padding to an index-word
boundary. The original widened per-edge type section is absent, not retained
beside a byte shadow. All index sections remain naturally aligned, including
odd and empty edge counts. Native consumers validate resource dtype, rank,
exact shape and overflow-safe byte counts before interpreting pointers.
Reorder/canonical-restore consumers reject this host-lifecycle representation.
Required CSC routing remains unchanged, with the cached Tensor retaining
ownership of the actual mixed-width allocation.

For index width ``w``, actual resource bytes are
``w*(E+S+1+(E+2*P if P else 0)) + E + ((-E) % w)``.
Saving against the lifecycle baseline is
``(w-1)*E - ((-E) % w)``, not simply three bytes per edge when a surface
requires tail padding. ``metadata_bytes`` and ``metadata_shape`` describe
the actual resource including that padding; shape counts index words.
``metadata_payload_bytes`` excludes padding, and ``metadata_padding_bytes``
reports it separately. ``metadata_dtype`` equals ``index_dtype``;
``type_dtype`` alone is uint8. ``type_byte_offset`` points into
``ReadVariableOp(..., dtype=index_dtype).numpy().view(numpy.uint8)``;
decode only the next ``E`` bytes as edge types. ``type_payload_bytes``
counts that section once. These fields are also returned symmetrically by
``csr_host_lifecycle_metadata()``. Deduplicate master/fused resource aliases;
payload bytes are not allocator capacity or driver reservations.

For resource-independent census, iterate
``connectivity["metadata_resource_specs"]`` (also available as the
``metadata_resource_specs`` property). Every owned replica resource has an
entry containing its existing ``handle``, true ``dtype``, allocation
``shape``, ``payload_bytes`` and ``role="csr_metadata"``. Read each resource
with its own declared dtype, not a dtype inferred from ``index_dtype``:
``ReadVariableOp(resource=spec["handle"], dtype=tf.as_dtype(spec["dtype"]))``.
The read shape and dtype-size product must equal ``payload_bytes``. This
field includes explicit alignment padding; ``logical_payload_bytes`` excludes
it and ``alignment_bytes`` reports it separately. ``schema``, ``type_dtype``,
``type_payload_bytes`` and exact ``type_logical_min``/``type_logical_max``
are included in each spec in homogeneous and typed modes. Empty type vectors
have null extrema, not a fabricated zero. Deduplicate by ``handle.ref()``.
Specs are creation-time Python receipts: accessing them performs no tensor
read, materialization, allocation, synchronization or host/device copy.
All prior singular fields and ``metadata_resource_handles`` remain available.

Both fat operators must be rebuilt. The existing native forward proxy,
weight-only VJP, direct-master spike VJP, direct-master carrier VJP and
direct-master combined sweep advertise a new default-false ``uint8_types``
attribute. Disabled Python calls omit it, retaining old-library call
signatures. Explicit typed construction checks all five capabilities and
errors before allocation with old/partial libraries. The attribute selects
the byte pointer reader; all accumulation arithmetic, queues, source splits,
pair metadata, CSC and launch attributes remain the existing implementation.
Actual graph attributes and independently decoded resource bytes, not the
report marker alone, establish execution. Metadata is derived at construction;
master/permutation checkpoint keys, Adam slots, canonical loss references and
canonical export boundaries are unchanged.

Adaptive routing
----------------

``use_adaptive_routing`` selects the ``adaptive_segment_ticket_v1`` forward sweep
schema on top of uint8 types; ``false`` keeps the dense tile sweep on the same
binaries. See :doc:`dpointnet_adaptive_routing`.

Ownership and native reads
--------------------------

``cell.recurrent_weight_values`` and every
``cell.inputs[name]["input_weight_values"]`` are actual live FP32 variables
in their respective source-major CSR order, including frozen named inputs.
Their compute attributes alias the exact master objects; the separate CSR
compute-shadow attributes are ``None``. There is no second persistent
canonical FP32 master, FP16 weight shadow or FP32 CSR copy.

The native ``ProjectionWeights<T, Index, true>`` type is ``const float*``.
Both joint projection and spike VJP specializations use ``weights[csr_edge]``
directly, independently of half/float spike and basis types. There is no
identity-edge permutation passed to an indirect reader. The optimized
combined/cooperative/fanout/gather arithmetic, queues, geometry and native
carrier accumulation remain the baseline mechanisms.

Native hooks are ``dpointnet_combined_sweep_csr_master``,
``dpointnet_csr_spike_grad_csr_master`` and
``dpointnet_csr_spike_grad_accumulate_csr_master``. The retained producer is
the separately named VJP-only proxy ``dpointnet_csr_spike_forward_csr_master``
(op type ``DpointnetCsrSpikeForwardCsrMaster``, FP32 ``weights``, optional
``uint8_types``); ``DpointnetCsrSpikeForward`` keeps its previous schema, so
GraphDefs and SavedModels serialized before CSR masters existed import unchanged.
The acceleration-report marker is ``optimized_combined_csr_fp32_master``.
These hooks prove selection, not executed-kernel coverage.

All dynamics gradients and Adam slots address live CSR variables. Recurrent
VJPs and carrier seeds stay CSR-ordered; the segmented variable transform is
an identity in this mode. The weight-only named-input native VJP uses
``DpointnetCsrWeightGrad(write_csr_weight_gradient=True)`` to write directly
into live CSR positions, including the packed batch-32 specialization. The
default remains canonical for the disabled baseline. Only the destination
address changes; arithmetic, traversal and launch geometry are unchanged.
No ``DpointnetCsrReorder`` or ``DpointnetCsrRestore`` is needed in a native
update. Both operators must be rebuilt for this attribute; an older
library is not sufficient.
Constraint signs and EMD connection-type groups are reordered with weights.
Frozen inputs remain live resource reads without optimizer participation.
Noise streams, Poisson count generation, delays and neuron dynamics are
unchanged. No trajectory-equivalence or convergence claim follows from this
precision/layout change.

External weight-loss ordering contract
--------------------------------------

``cell.weight_loss_values(name="<recurrent>")`` returns the live optimizer
variable itself, not a canonical copy. All per-edge references, group labels,
and indices consumed by a loss must address the same live order. Dynamics,
event-weight VJPs, regularizer gradients and Adam slots are all in this order.
The training engine does not canonical-restore gradients for CSR-live masters:
segmented replay has no variable-gradient transform, and full BPTT uses an
identity boundary before adding any outside regularizer gradients.

For an external EMD constructor that builds canonical group indices and a
sorted initial distribution from the original SONATA network, compose its
existing gather indices once, before tracing or timing::

    emd = ExternalEMD(strength, canonical_network, dtype=tf.float32)
    emd._group_order.assign(cell.weight_loss_indices(emd._group_order))
    # In the timed loss:
    value = emd(cell.weight_loss_values())

This explicit adapter targets the external constructor's existing integer
group-order variable. It preserves its already-sorted initial values and
group slices: the same canonical group rows are fetched directly from their
live CSR positions. Do not also permute sorted initial references. There is
no extra per-update whole-edge canonicalization gather; only the original
EMD gather remains. Recreate the derived group indices from canonical metadata
before adapting; do not compose an already-adapted order a second time.

An external constructor that instead accepts per-edge reference arrays or
group labels can prepare those once with
``cell.weight_loss_reference(canonical_array, name=...)``. This does not
change units: normalized weights, physical SONATA weights and sorted
group-reference distributions are different contracts. Never permute only
the weights while leaving per-edge group metadata canonical.

``cell.canonical_weight_values()`` is appropriate for untimed logical master
hashes, export and migration. Record raw physical-master hashes separately.
Canonical-order model hashes must match CSR-master logical canonical hashes,
not raw CSR hashes. Calling the canonical accessor inside every
timed EMD evaluation would add a whole-edge gather and is not the preferred
adapter above.

Canonical boundary and checkpoint conversion
--------------------------------------------

``cell.recurrent_master_layout`` and
``cell.inputs[name]["master_layout"]`` expose reversible
``csr_to_canonical`` and ``canonical_to_csr`` permutations. Duplicate endpoint
rows remain independent; the actual connectivity builder defines their
stable order. ``layout.to_live(values)`` gathers canonical rows into CSR;
``layout.to_canonical(values)`` performs the inverse. They also apply to
per-edge optimizer slots and constraint arrays.

Use ``cell.canonical_weight_values(name="<recurrent>")`` for an on-demand
canonical tensor and ``cell.assign_canonical_weights(values, name=...)``
for canonical assignment. Native projection never calls these adapters.
``ModelWeights`` uses canonical values before applying canonical physical
export factors and SONATA edge ordering. A deep copy snapshots canonical host
rows at construction; a live, non-deep-copy export keeps only references to the
CSR-ordered masters and converts them on each ``to_dataframe()``. Direct
``master.assign`` explicitly means CSR order.

Checkpoints use the canonical format. Each CSR-ordered master and endpoint index
variable, and every optimizer slot paired with a master, serializes a
host-converted canonical copy under the ordinary ``VARIABLE_VALUE`` key and
converts restored canonical rows back to the live layout; the live variables are
never reordered. Compute-shadow entries that canonical-order models track
(``recurrent_weight_values_compute`` and, with native CUDA,
``recurrent_csr_weight_values_compute``) are written by checkpoint-only stand-ins
holding the cast master values and are ignored on restore. Checkpoint object
graphs and values are therefore identical to those of a canonical-order model with
the same configuration, so ``tf.train.Checkpoint`` files restore in both
directions (including checkpoints written before this option existed) and
``assert_consumed()`` holds. Optimizer state is paired explicitly, never by shape,
through the optimizer's own variable index and the complete per-variable slot lists
of an allow-list (``csr_master.OPTIMIZER_SLOT_ATTRIBUTES``; Keras 3 and TF-Keras 2
share the class names):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Optimizer
     - Per-variable slot lists
   * - Adam, AdamW, ``exp_adam`` (``ExponentiatedAdam``), with or without AMSGrad
     - ``_momentums``, ``_velocities``, ``_velocity_hats``
   * - SGD (``sgd``; momentum, Nesterov)
     - ``momentums`` (empty without momentum)
   * - RMSprop (momentum, centered)
     - ``_velocities``, ``_momentums``, ``_average_gradients``
   * - Adagrad
     - ``_accumulators``
   * - Adamax
     - ``_m``, ``_u``
   * - Nadam
     - ``_momentums``, ``_velocities``
   * - Adadelta
     - ``_accumulated_grads``, ``_accumulated_delta_vars``
   * - Ftrl
     - ``_accumulators``, ``_linears``
   * - Lion
     - ``_momentums`` (Keras 3) or ``momentums`` (TF-Keras 2)
   * - Lamb (Keras 3)
     - ``_momentums``, ``_velocities``
   * - Every class above
     - ``_model_variables_moving_average`` (``use_ema``),
       ``_accumulated_gradients`` (Keras 3 ``gradient_accumulation_steps``)

Subclasses inherit their base's entry. ``LossScaleOptimizer`` wrappers (Keras 3 and
TF-Keras 2) are paired through their inner optimizer, and TF-Keras 2 legacy
optimizers through ``get_slot_names``/``get_slot``. Binding raises for any other
optimizer class that tracks a CSR master, and for any variable saved under the
optimizer that is shaped like a CSR master but is not paired (for example extra
state added by a subclass); use ``use_csr_master_weights=false`` with such
optimizers. ``RNN.train`` and ``TrainingEngine`` (``train``, ``train_step``) build
the optimizer and call ``cell.bind_optimizer_checkpoint_slots(optimizer)`` before
the first update; custom loops must call it after building the optimizer and
before its first update. The restore order does not matter: slots that a
checkpoint restored before binding (TensorFlow applies a pending restore when
``optimizer.build`` creates them, and records the restore on the slot) hold
canonical rows and are converted to the live layout once. With
``after_update=True`` (used after an update that built the optimizer) such slots
raise instead, because the update may already have used them. Binding also raises
when a checkpoint restored the optimizer's state (slots, iterations or other
optimizer variables, so also for slotless optimizers such as SGD without
momentum) but not a master the optimizer updates: Keras 3 checkpoints trainable
named-input masters only through the optimizer's variable list, which
``optimizer.build`` replaces, so restore after ``TrainingEngine.build_optimizer()``.
Endpoint restores verify that the stored canonical connectivity equals the
model's, in eager and graph mode, and never assign the endpoint variable: a
mismatch raises ``ValueError`` eagerly and ``InvalidArgumentError`` from a
graph-mode restore op. Keras weight files
(``save_own_variables``/``load_own_variables``) and the EMD regularizer's group
order store canonical rows as well Keras 3 weight files are interchangeable
between layouts; TF-Keras 2 weight files list canonical models' compute shadows
positionally, so cross-layout loads there are rejected by the variable count.
Under TF-Keras 2 the tracked ``inputs`` dictionary also receives per-input shadow
stand-ins, and restore lookups resolve them through ``_lookup_dependency``. SavedModel export and graph-mode checkpoint
saving use a CPU permutation instead of the host path.

``bmtk.simulator.dpointnet.csr_master.transfer_canonical_state`` remains available
to copy state between two live cells explicitly, for example between different
optimizer instances, without writing a checkpoint:

#. Construct and restore the source cell with its original representation.
#. Construct the target with the same canonical graph, initialization, variable
   dtype and constraints, selecting its target representation before construction.
#. Build both optimizers. Supply an explicit list of ``(source_slot,
   target_slot)`` pairs for every surface to
   ``bmtk.simulator.dpointnet.csr_master.transfer_canonical_state`` as
   ``per_edge_slots``. Include frozen surfaces with an empty list.
#. Supply optimizer iteration/learning-rate and other nonedge state through
   ``scalar_state``. The helper converts masters and all supplied slots through
   canonical order, validates shapes and refreshes baseline shadows.
#. Save a new target checkpoint explicitly tracking every master, including
   frozen named inputs, the optimizer, noise and other runner state. Restore
   the new checkpoint in the target representation and destructively verify
   each resource.

The helper does not infer optimizer-slot ownership from variable names.
The caller must provide every slot and independently copy nonoptimizer
runner/noise/loss state; it is not an automatic arbitrary-checkpoint converter.
Checkpoint mappings are reconstructed from the identical graph; never restore
against a different connectivity ordering.

Build and qualification boundary
--------------------------------

Build both operators with GPUs hidden from the declared TF2.21 prefix::

    CUDA_VISIBLE_DEVICES="" DPOINTNET_CUDA_ARCHS="61 70 75 80 86 89 90" \
      python -m bmtk.simulator.dpointnet.custom_ops.build

Both manifests and binary hashes must be recorded. SM61/70/75/80/86/89/90
and PTX90 compilation is not execution qualification on every GPU family.
Focused tests cover canonical/CSR conversion, duplicate rows, canonical
checkpoint format, checkpoint restore in both directions with Adam slots,
explicit Adam migration, projection/VJPs,
named trainable/frozen masters, reused compiled graphs, assignment, checkpoint
restore and full/segmented-remainder replay. Full-network timings and memory
controls belong to the parent orchestration, not these component tests.
