Private CSR-ordered FP32 live-master prototype
=============================================

This isolated, uninstalled snapshot is not a release or default change.
Set ``_DPOINTNET_EXPERIMENTAL_CSR_FP32_MASTERS=1`` before constructing
``GLIF3Cell``. Unset it or set it to ``0`` for the same-source shadow
baseline. The earlier ``_DPOINTNET_EXPERIMENTAL_MASTER_PROJECTION`` flag
does not activate this prototype.

The admitted execution is single-replica NEST, independently parameterized
FP32 edge weights, spike/count inputs, compute temporal gradients and the
optimized native combined sweep. CPU primitive probes are correctness
references only. This is not FP32-temporal, distributed, local-learning,
continuous-current or arbitrary saved-projection support.

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
``dpointnet_csr_spike_forward(csr_master_weights=True, vjp_only=True)``.
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
candidate update. Both operators must be rebuilt for this attribute; an older
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
Baseline canonical hashes must match candidate logical canonical hashes,
not candidate raw CSR hashes. Calling the canonical accessor inside every
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
export factors and SONATA edge ordering. A live, non-deep-copy export rereads
masters each time. Direct ``master.assign`` explicitly means CSR order.

Raw cross-layout checkpoint restore is intentionally rejected: CSR resources
serialize a ``CSR_MASTER_V1_<permutation SHA256>`` key rather than the legacy
``VARIABLE_VALUE``. Different CSR permutations also reject raw restore,
including CPU reference versus GPU target-sorted layouts.
Automatic legacy restore is unsafe even when variable shapes agree.
The cell also adds a nontrainable int32 layout-version scalar, so legacy
Keras topology weight counts cannot silently match. Only explicit TensorFlow
resource checkpoints are qualified here; Keras/HDF5 round trips and restores
with skipped mismatches are unsupported.
Use the following explicit conversion protocol in either direction:

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
Focused tests cover canonical/CSR conversion, duplicate rows, raw-format
rejection, explicit Adam migration in both directions, projection/VJPs,
named trainable/frozen masters, reused compiled graphs, assignment, checkpoint
restore and full/segmented-remainder replay. Full-network timings and memory
controls belong to the parent orchestration, not these component tests.
