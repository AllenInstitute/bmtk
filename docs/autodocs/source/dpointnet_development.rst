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
gradients. Type-indexed state dispatch checks current coefficient identity.
Static dispatch is an explicit promise that within-type equality remains valid,
not a mutation guard or a cache of initial values. Uniform type-preserving
changes need not invalidate that representation.

The single-replica SM86/TF2.21 high-level automatic route can move the live
identity predicate out of native custom-gradient carrier arguments. Both branches reuse
the same neuron transition/VJP; assignment and restore still affect the next
rollout. Custom cells and distribution retain their existing dispatch.

Custom runners and FP32 temporal gradients
------------------------------------------

External current and release components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``GLIF3Cell.advance_with_currents(inputs, states, currents, noise_seed=...)``
advances the existing neuron transition without rerunning recurrent or input
projection. Supply the complete **base** carry (including input-delay and online
loss channels), not appended STP release state. Currents have shape
``[batch, n_neurons * n_syn_basis]``, already include normal learning-rate scaling,
and are cast to the selected compute dtype as in ordinary execution. Inputs are
still required for external delay history and replay-safe Poisson history.
``noise_seed`` is the scalar base seed, not a two-element stateless RNG seed.
The method deliberately bypasses a subclass's ``_call_impl`` projection override;
the subclass remains responsible for its own release-state update and validation.

The hook preserves ordinary-tape current/state VJPs and can be used inside
segmented recomputation. It rejects the FP32 temporal-adjoint mode just as direct
``cell.call`` does; it is not an external-runner canonical-gradient boundary.
Existing ordinary calls, state ordering, precision and accelerator selection are
unchanged.

``cell.native_execution_options`` returns a copy of the base cell's selected
fused-current/state/history, pair-projection and fused-event flags, including
when the profile is explicit and ``acceleration_report`` is ``None``.
It is selection information, not kernel-execution evidence or a declaration
about a custom projection. Mutating the returned dictionary changes no policy.
``cell.create_compute_shadow(value, name=..., dtype=...)`` creates a nontrainable
copy without creating a canonical master or registering gradient routing.
Its default dtype is the cell compute dtype; FP32 projection adapters must
request FP32 explicitly. The owner must refresh it after every master update,
assignment and restore, including frozen assignable masters, and explicitly
verify checkpoint coverage. A shadow factory is not automatic synchronization.

``fused_spike_currents`` accepts graded **nonnegative** source amplitudes.
It multiplies positive amplitudes unchanged, including fractional release
efficacy and counted Poisson events. With spike gradients enabled its VJP
returns dense source-row credit, including silent connected rows. For STP that
is efficacy credit: apply the local release VJP before summing source-event
credit across consuming ranks. Do not route efficacy credit directly to the
source spike surrogate.
With ``write_csr_weight_gradient=False`` the weight VJP is already canonical,
including duplicate edges, and must not receive a second CSR restoration.
An actual direct-CSR carrier still requires exactly one canonical restoration.

Temporal adjoint boundary
~~~~~~~~~~~~~~~~~~~~~~~~~

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

Current accumulation and numerical boundaries
---------------------------------------------

``fused_combined_currents`` is a distinct projection primal: compatible
recurrent/named-input sources share FP32 accumulation and one final rounding.
It retains each source's dense source VJP and canonical/carrier weight boundary;
``vjp_only`` projection proxies supply derivatives, not additional currents.
Never apply a second canonical restoration to an already-canonical VJP.
The tile attribute requires the rebuilt native signature; an older library
cannot silently accept it. Routing caches depend only on immutable metadata
identity and source width, never the current weight values.
See :doc:`dpointnet_performance` for automatic admission and numerical opt-out.

``fused_spike_currents(initial_currents=...)`` rounds the standalone projection
to the compute dtype before adding the initial current once. Do not seed the
projection's atomic accumulator or fixed-four reduction with the initial
current: that reassociates every synaptic addition with the baseline, can
absorb small contributions, and changes threshold crossings even in FP32.
The fixed-four route may forward its initial buffer because each thread reads
and writes only its own output elements. Atomic projection routes retain a
separate output until projection completes, then add the initial current on
the same TensorFlow device stream. The initial-current VJP is the identity;
spike and canonical weight VJPs are unchanged.

This rounding boundary does not make atomic projection deterministic or
make differently grouped sums of multiple projections bitwise equivalent.
Tiny current differences can amplify through recurrent spike thresholds.
Validate per-neuron counts and states, not only aggregate spike totals;
capture forward results before backward/update when diagnosing divergence.
For checkpoint replay requirements, retain the recorded-primal guidance above.
``use_fused_current_accumulation`` concerns forward currents and is independent
of ``use_fused_recurrent_accumulation``, the recurrent weight-gradient carrier.

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
LGN filtering on GPUs with compute capability 7.0 or earlier uses bounded
extract-patches/matrix multiplication,
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
