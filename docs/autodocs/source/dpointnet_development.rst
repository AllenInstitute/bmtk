DPointNet contributor reference
===============================

This page is for contributors and authors of custom execution loops, not a
first-run configuration recipe.
Current user guidance lives in :doc:`dpointnet_guide`,
:doc:`dpointnet_training_standards` and :doc:`dpointnet_performance`.

GLIF3Cell gradient-control API
----------------------------------------

The constructor retains old positional argument slots for compatibility.
Its parameter documentation marks deprecated aliases and their replacements;
the :ref:`migration table <dpointnet_training_standards--clear-names-for-gradient-controls>`
also covers JSON configurations.

.. py:class:: bmtk.simulator.dpointnet.cell_models.glif3_cell.GLIF3Cell
   :noindex:

   :param spike_surrogate: Preferred selector: ``"gaussian"`` (default) or ``"triangular"``.
   :param spike_surrogate_width: Preferred Gaussian width (default ``0.28``).
   :param spike_surrogate_gain: Preferred surrogate gain (default ``0.05``).
   :param recurrent_spike_gradient_scale: Preferred recurrent gradient retention (default ``1.0``).
   :param voltage_state_gradient_scale: Preferred voltage gradient retention in [0, 1] (default ``1.0``).
   :param gauss_std: Deprecated compatibility alias for ``spike_surrogate_width``.
   :param dampening_factor: Deprecated compatibility alias for ``spike_surrogate_gain``.
   :param recurrent_dampening_factor: Deprecated compatibility alias for ``recurrent_spike_gradient_scale``.
   :param voltage_gradient_dampening: Deprecated removed fraction; convert to ``voltage_state_gradient_scale = 1 - clip(value, 0, 1)``.
   :param pseudo_gauss: Deprecated Boolean selector; True maps to ``"gaussian"``, False to ``"triangular"``.

   Explicit aliases emit ``FutureWarning``; omitted aliases do not warn.
   Existing values and positional slots remain supported, with no scheduled
   removal version. Supplying both names for one control raises ``ValueError``.

Terminology
-----------

* **Canonical weights:** trainable variables in SONATA edge-row order.
* **CSR:** compressed sparse row, an internal connectivity layout grouped
  by source neuron rather than original edge order.
* **Compute shadow:** a refreshed, possibly lower-precision copy of a master
  weight used by the forward computation.
* **Weight carrier:** a differentiable loop input whose backward value
  accumulates weight derivatives; an FP32 carrier does not imply FP32
  spike or current derivatives.
* **VJP:** vector-Jacobian product, the backward propagation of a supplied
  output derivative through an operation.

Canonical weights and gradients
-------------------------------

Preserve SONATA neuron and edge-row identity. Trainable FP32 masters, optimizer
slots, constraints, checkpoints, regularizers and exports remain canonical;
source-CSR shadows are internal compute representations.
Restore CSR dynamics gradients exactly once at their enclosing training boundary
before composing canonical regularizer gradients or applying an optimizer.
Match distributed masters and replica components by exact identity, never names.

Refresh recurrent and all trainable named-input ordinary/CSR shadows after updates,
explicit assignment, restore or local plasticity. Frozen masters must not leave
stale nonzero compute shadows. Preserve Dale's law, configured bounds and
duplicate-edge ordering; do not prune silent-neuron spike adjoints.
Preserve Poisson counts, invocation-local replay/RNG state, external delay history
and NEST end-of-step timestamps.

Automatic direct-CSR gradients require a training-engine or FP32-temporal
restoration boundary. Bare cells and inference-only RNNs retain canonical
gradients. Type-indexed state dispatch checks current coefficient identity;
static dispatch is an explicit assumption that those coefficients cannot change.

Custom runners and FP32 temporal gradients
------------------------------------------

The supported FP32 temporal route uses ``TemporalAdjointRunner`` or
``ExplicitStateRNN`` with selective forward state. The reverse pass uses
FP32 state/projection derivatives and can record the original compute-precision
currents to preserve the primal trajectory. ``recompute`` is approximate when
atomic projections are nondeterministic.

Do not wrap this runner in a second segmented gradient runner or restore CSR
order twice. A private ``tf.while_loop`` with FP32 weight carriers is not
automatically this route: inspect every state/current/spike derivative dtype.
``fused_recurrent_weight_carry(vjp_only=True)`` requires FP32 operands;
ordinary direct-loop use supplies compute-dtype primal currents.
External custom runners must wire their input and recurrent carrier surfaces.
Explicit packed/small-batch FP16 backward requests are incompatible with the
FP32 temporal route.

Online loss channels
--------------------

Register state-dependent loss channels before model construction. Retain FP32
online accumulators under mixed precision and preserve the chosen physical-state
policy. For ``VoltageRateFloor``, preserve pre-reset voltage semantics and
refractory gating without changing spikes or reset/ASC derivatives.

Rate-history updates happen only after accepted optimizer updates, never during
validation, replay or a rejected loss-scaled step. Pool rates across the actual
combined batch; series and accumulated updates have different commit boundaries.
Serialize history and reconstruct the same channel configuration before restore.
The initial gate is inactive until measured rates are committed or explicitly
initialized, not an assumed all-silent population.

Connectivity preprocessing
--------------------------

``build_csr_connectivity`` deduplicates compact ``(post, synapse_type)`` pairs
with uint64 keys when ranges fit. The original tuple path handles
unrepresentable ranges/strides.
Stable packed ``(source, target, synapse_type)`` sorting preserves the original
lexicographic permutation, including duplicates.
``lex_sort_order_np`` similarly packs nonnegative integer pairs; signed, floating,
empty or overflowing cases retain the original lexsort.

Use unsigned integer arithmetic, never float64 keys or approximate ordering.
Compare packed-key paths with independent tuple/lexsort references and test
overflow, signed, floating and empty cases. These optimizations require no user
configuration or persistent cache and do not change model/RNG initialization.

Compatibility implementation
----------------------------

Below compute capability 7.0, scalar half accumulation uses TensorFlow's
compare-and-swap helper and duplicate-target warp grouping uses shuffle/ballot
matching. Preserve grouping and live silent-neuron derivatives.
Pascal/Volta LGN filtering uses bounded extract-patches/matrix multiplication,
retaining padding, normalization and multichannel temporal filtering.
Do not generalize that input-specific route into a blanket convolution fallback.

Alpha-basis fitting uses shared time constants and unconstrained per-class
least squares. Its objective is the unweighted mean waveform error; acceptance
uses the maximum per-class relative RMS error. Preserve distinct kinetic-class
identity, supplied coefficients and metadata. Automatic tolerance relaxation,
amplitude renormalization, nonnegative coefficient constraints and
neuron-frequency weighting would change the fitting method.

Validation and attribution
--------------------------

Test independent values and VJPs, canonical ordering, live coefficients, distributed
identity, replay, state/delay/RNG carry and strict model/optimizer restore.
Keras3 success does not replace actual Keras2 symbolic-state testing.
Rebuild both architecture-covered operators after CUDA/signature/ABI edits.
Do not relax tolerances, precision, clipping or scientific constraints to pass.

Variable-batch tiles, scaled-half projection and sparse weight accumulation adapt
Javier Galvan's ``JavierGalvan9/V1_GLIF_model`` implementation; retain the source
attribution and exact revision comments in the corresponding code.

Alpha fitting and drifting-grating input placement also draw on
``JavierGalvan9/V1_GLIF_model`` at
``2c52ec10c1eee409ddf900f8a5b8460cf9c46d24``. The alpha fitting reference is
``synaptic_data/alpha_basis_calculation.ipynb``. Preserve attribution while
distinguishing generalized network-fitting bounds and DPointNet's seeded
broadcast trial assignment from that reference implementation.
