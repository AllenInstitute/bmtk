DPointNet contributor reference
===============================

This page is for contributors and authors of custom execution loops, not a
first-run configuration recipe.
Current user guidance lives in :doc:`dpointnet_guide`,
:doc:`dpointnet_training_standards` and :doc:`dpointnet_performance`.

.. toctree::
   :maxdepth: 1

   dpointnet_csr_master_prototype
   dpointnet_adaptive_routing

Running tests efficiently
-------------------------

Install the test dependencies with ``python -m pip install -r test_requirements.txt``.
For CPU-only development, use four isolated pytest workers and one numerical
thread per worker::

    CUDA_VISIBLE_DEVICES="" BMTK_DPOINTNET_DISABLE_FUSED_CUDA=1 \
      TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python -m pytest -q -n 4 --dist worksteal --durations=20 \
      tests/simulator/dpointnet

Use fewer workers on hosts with fewer CPU cores or limited memory. Work stealing
balances individual parameterized cases, including expensive traced replay and
adjoint oracles. It does not select a reduced subset. Run focused files or node
IDs during iteration; omit ``-n 4`` for serial debugging.

The pull-request workflow retains Python3.8,3.9,3.10 and3.11 compatibility
jobs. Each runs the complete DPointNet suite with four CPU workers; other
simulators, builders and utilities remain serial because some use shared
files or simulator-global state. Coverage is collected only on Python3.11,
appended across both phases and uploaded as ``coverage.xml``. Dependency
downloads are cached; a newer commit cancels the obsolete run for the same PR.
Test-only and test-dependency changes also trigger validation.

These CPU runs deliberately disable native CUDA loading. They do not replace
serial native-GPU qualification from the intended source and matching operators.
Do not run multiple pytest GPU workers against one GPU. Supported deprecated
aliases retain regression coverage; a deprecation warning is not evidence that
an API or its tests can be deleted.

The compiled firing-rate oracle groups silent, binary and count inputs in one
trace per units/length/dtype combination, keeping separate losses and gradients
for all three inputs. This removes redundant trace setup, not numerical cases
or their exact-equality assertions.

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

Cooperative short-row projection
--------------------------------

The keyword-only ``use_cooperative_row_projection`` option of ``GLIF3Cell``
(also forwarded through ``rnn_cell_params``) requires the explicit/selected
combined current sweep and both rebuilt native libraries. An explicit request
errors on an older native ABI; the wrapper omits the new native attribute when
the option is off, retaining compatible older-library behavior. Automatic
selection enables it whenever the joint sweep is admitted and the capability is
present; explicit ``False`` keeps the original traversal.

The native ``cooperative_rows`` list is empty by default. The wrapper opts in
only known target-unique CSR sources below512edges per delay-expanded source
row; repeated-target/high-fanout named sources use the fanout-row walk below
when it is selected, and otherwise their original traversal.
Eight short row segments are packed by warp scans into a register-only work
stream. Per-source float products, FP32 tile accumulation and final rounding
remain unchanged, but atomic addition order can differ. Canonical/master and
carrier VJPs use the unchanged per-source proxies, including silent credit.
Counts, delays, live weights, coefficient dispatch, Poisson RNG and state/
checkpoint schemas are unchanged; active queues are still rebuilt.

Compile-time default/opt-in specializations plus two-block launch bounds avoid
the first prototype's register-pressure regression. Measured RTX3090 B32 mixed
full-biological-loss controls give about4.17percent shorter updates. Neither
that hardware result nor fat-binary compilation establishes every-family
performance, convergence or distributed/custom-cell composition.

High-fanout compact row walk
----------------------------

The keyword-only ``use_fanout_row_projection`` cell option, also accepted
through ``rnn_cell_params``, requires the combined sweep and the rebuilt native
``fanout_rows`` capability. Explicit incompatible requests raise; when off the
wrapper omits the new native attribute for older compatible libraries.
Automatic selection enables it with the admitted joint sweep; explicit
``False`` keeps the original traversal. It is independent of the cooperative
short-row selector.

Sources with average fanout at least512edges per delay-expanded row are walked
as follows. One packed block scan over the round's active rows compacts only
the rows with a nonempty segment in the block's target tile, in queue order,
and yields their edge prefix. Each warp then walks32consecutive edges of the
round's concatenated segments: the chunk owner is found by one warp-uniform
binary search and each lane advances linearly where its edge crosses a row.
No lanes are padded at row ends. Equal-target runs inside a row segment are
detected by comparing each lane's key with its predecessor (segments are
target-sorted), and suffix rounds stop as soon as no lane can still take from
its run; this replaced a per-edge warp-wide value match and fixed five rounds,
and applies to every high-fanout traversal of the combined sweep.
Row-local repeated-target aggregation, canonical per-source VJPs, silent
credit, FP32 products/tile sums, live operands, delays and Poisson counts are
preserved. Atomic ordering can change; there is no bitwise trajectory/
convergence guarantee or new state ABI.

Tiles wider than2048targets now launch1024-thread blocks when the index type
is32-bit, so one wide-tile block fills an SM instead of leaving half of it
idle; both block scans use warp-scan scratch to fit the row tables beside a
5120-target tile. Explicit ``combined_current_target_tile_size=5120`` is how
this geometry is selected; the automatic default tile is unchanged.
Earlier conclusions that wide tiles regress this workload were measured with
512-thread blocks and do not transfer to the1024-thread kernel.

Identical-operand RTX3090 B32 mixed forward screens (fixed captured LGN step,
synthetic recurrent activity, recurrent plus LGN only) measure the joint main
kernel at549us with tile5120 versus575us at tile2048 and the retained Javier
original at544us; the LGN-only kernel is261us versus252us. FP32 joint is551us
versus Javier579us. With the background source included, tile5120 is slower
than2048 (646 versus627us), and full-biological B32 updates at tile5120 were
2.1percent longer than at2048; the wide tile is not a general recommendation.
At tile2048 the compact walk and run detection alone shorten full-biological
updates from2.6155to2.5779seconds (1.44percent).
The remaining per-step difference outside the sweep is queue construction: the
sweep rescans every delay slot each step, whereas carried per-slot queues would
scan only the newest slot. These are not parity, memory-saving, FP32
full-training admission or automatic recommendations, and compiled fat-binary
targets are not execution evidence.

Target-major gather for low in-degree sources
---------------------------------------------

The keyword-only ``use_target_gather_projection`` cell option, also
accepted through ``rnn_cell_params``, requires the combined sweep and the
rebuilt native ``gather_sources`` capability; explicit incompatible requests
raise and the wrapper omits the attribute for older compatible libraries when
the option is off. Automatic selection enables it with the admitted joint
sweep; explicit ``False`` keeps the scatter traversal.
Sources with at most16incoming edges per target on average
(``combined_gather_eligible``) are projected target-major: the operator builds
once per metadata identity a stable target sort of the edge positions with
per-target ranges and per-entry source rows, and after the scatter phases each
tile thread sums its own targets' few incoming edges in FP32 (four independent
chains at a time) and adds once to the shared tile without atomics, row lookups
or scans. Repeated sources on one target are summed like any other entry.
Canonical per-source VJPs, live weights, delays and Poisson counts are
unchanged; the single rounding of the tile is preserved.

This exists because the V1 background input (100sources, four edges per neuron,
250Hz Poisson counts) is a poor fit for the source-major scatter: its sweep
phase cost about123us per launch for under two million edges, dominated by
per-block fixed work and shared CAS-loop atomics. The gather measures about82us
in the three-source screen and adds about10to25us when other sources run.
RTX3090 B32 mixed full-biological updates fall2.5779to2.5558seconds
(0.85percent) on top of the compact walk, 2.2percent below the LGN-winner
baseline. The structurally better placement, projecting such an input inside
the neuron update as the original implementation does, is not attempted here.
Three earlier gather variants (one sequential target per thread; entry-parallel
with atomics; warp-owned pipelined ranges) measured105,133and141us and were
rejected: any in-sweep gather rereads the source's target-major table once per
(tile, sample) block. The automatic policy enables the gather with the
admitted joint sweep; its eligibility rule (at most16incoming edges per target
on average) is a workload property, and explicit ``False`` retains the scatter.

Prepared CPU Poisson draws
--------------------------

The keyword-only ``use_prepared_cpu_poisson`` option of ``GLIF3Cell`` is
also passed through high-level ``rnn_cell_params``. Automatic selection
resolves it to ``"auto"`` for the single-replica NEST direct loop with compute
temporal gradients and internal Poisson inputs; the RNN layer then prepares
counts only when it actually runs the direct loop and the declared batch,
rollout length and input widths fit the 64 MiB count table, and otherwise
keeps the per-step sampler. Explicit ``True`` keeps the strict behavior below.
It requires the direct, forward, unmasked, non-unrolled, batch-major NEST loop
with compute-temporal gradients; FP32 temporal gradients and device Poisson
are explicitly incompatible. Stateful execution remains unsupported.

When enabled, each rollout/checkpoint chunk calls the unchanged CPU stateless
Poisson sampler with its original per-step counter, live rollout seed and
replica seed offset, then uploads its complete count table. Counts are stored
losslessly in int64 because TensorFlow 2.21 places int32 ``GatherV2`` parameters
on the host. A separate GPU int64 loop index selects each frame, cast back to
int32 before the ordinary cell input path. Do not replace this with int32
table gathers: actual traces show a full-table upload every timestep.

The option bounds resident tables to 64 MiB per chunk, across named internal
Poisson/noisy-current inputs. It does not cache counts across rollouts, advance
the logical noise stream, change the noise state/checkpoint schema or replace
Poisson counts with Bernoulli events. Recomputed chunks prepare the same draws
from their restored starting counter. A step-level seed override while using
a prepared frame errors. Direct standalone cell calls retain ordinary sampling.

RTX3090 B32/T500/checkpoint25 full-biological-loss controls show about
1.8 percent shorter updates, without a reproducible memory saving.
Distributed/custom-cell composition, other GPU families and full-loss FP32
admission are not qualified by this change.

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

The single-replica SM86/TF2.21 high-level automatic route moves the live
identity predicate out of native custom-gradient carrier arguments. With rebuilt
device-dispatch entry points, the state kernels read that GPU predicate directly:
true selects the compact type table, while false selects the live full AoS
coefficient table. Forward, history and event/state VJPs share the existing
arithmetic. Neither branch requires a per-step host predicate transfer.
Preparation still checks current coefficients at every rollout boundary;
assignment and restore affect the next rollout without retracing. This is not
the static immutable-coefficient promise.

All six device-dispatch entry points must be available before selecting this
route. Older compatible libraries retain the existing TensorFlow conditional
and its full-coefficient fallback. Custom cells, distribution and explicit
static dispatch retain their existing selection boundaries. Rebuild both native
operators after changing the native ABI; old entry-point signatures are retained.

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
