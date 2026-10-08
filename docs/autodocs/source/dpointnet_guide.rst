#########
DPointNet
#########

DPointNet is a BMTK engine for the training and simulation of large-scale biorealistic neuronal circuits 
utilizing deep-learning techniques. Users can take a novel or existing network and train and save synpatic 
weights using pre-determined inputs and expected outputs. It can also be used for purely inference/simulation
of large scale networks often in a way that is much faster than other simulators.

For more information see paper on using software for building and simulation of cortical circuit model:

`Ito et. al., 2026 <https://www.biorxiv.org/content/10.64898/2026.03.13.711751v1>`_



Installation
============

DPointNet must be installed using the base bmtk package as seen in :doc:`the instllation guid <installation>`.

Besides the base dependencies, it requires tensorflow 2.14 to 2.16, and may work with gpu or without. To install
with gpu run the following in your environment:

::

    $ pip install tensorflow[and-cuda]

or without a gpu:

::

    $ pip install tensorflow

Recommended portable acceleration
---------------------------------

For new projects using the consolidated DPointNet fork, build both CUDA
operators for the full supported GPU pool and use automatic accelerator
selection instead of maintaining GPU-specific flag lists. Pin the qualified
fork revision ``94f90d70e6353af54bf6205708a7e2b75e85a534`` from
``shixnya/bmtk:feature/dpointnet-training-inputs-consolidated`` in the
project's dependency contract and verify the imported BMTK path. This is an
engineering preview, not an upstream release or an upgrade of existing
installed environments.

In the declared TensorFlow/CUDA environment, first check that
``nvcc --list-gpu-code`` supports all requested targets, then run:

.. code-block:: bash

  DPOINTNET_CUDA_ARCHS="61 70 75 80 86 89 90" \
    python -m bmtk.simulator.dpointnet.custom_ops.build

This includes Pascal, V100, RTX8000, A100, RTX3090, L40S and H200, plus SM90
PTX. Compilation on HPC belongs in an allocated compute job. Do not silently
drop unsupported targets; report a compiler mismatch. Binaries are specific
to the execution environment's TensorFlow/CUDA ABI.

The initial full-target build of both operators took approximately 13-14 minutes
on the tested 8-CPU HPC allocation, estimated from job-phase timings. This excludes
environment setup, queue waiting and model initialization/first-step tracing.
Reuse the binaries across supported GPUs and batches in a compatible environment;
rebuild after CUDA-source, TensorFlow ABI or CUDA-toolchain changes.

Add to the otherwise unchanged simulation configuration:

.. code-block:: json

  {
    "rnn_cell_params": {
      "acceleration_profile": "auto"
    }
  }

For eligible direct-loop BPTT, explicitly select
``use_direct_state_rnn_loop=true`` as well. Native recurrent accumulation
requires direct CSR, four bases and trainable per-edge weights; retain
per-type training if that is your scientific recipe. Inspect
``rnn.acceleration_report`` and qualify actual updates, checkpoint restoration
and memory on the intended GPU. Explicit individual flags override the profile.
External runners must wire the resolver and their independent input surfaces.

Automatic selection chooses qualified paths using actual GPU capability,
loaded operators, precision, topology and batch, not CUDA version alone.
RTX8000 ordinary FP16 backward and existing SM86+ native paths are admitted;
Pascal native paths remain explicit opt-ins, V100/A100 automatic paths remain
conservative, and RTX8000 FP32 compute/replay stays generic. Packed backward
remains batch32-only; batch16 retains eligible native accumulation.

This does not automatically choose batch, precision, dynamics, parameter
sharing, sampler, loss, optimizer or allocator. Fat compilation is not
universal runtime qualification or execution authorization. Preserve running
and immutable experiment pins. See
`performance configuration and qualification limits <https://github.com/shixnya/bmtk/blob/feature/dpointnet-training-inputs-consolidated/docs/dpointnet_parity.md>`_.

GLIF dynamics and explicit state
-------------------------------

``GLIF3Cell`` defaults to ``dynamics_mode="legacy"`` to preserve existing
trajectories and checkpoints. Legacy remains the throughput-oriented starting
point. NEST is currently an opt-in compatibility mode for NEST-aligned dynamics
and validation, not a change to the default or a performance optimization.

.. warning::

  **NEST training is experimental and is not recommended for use.** Use
  ``dynamics_mode="legacy"`` for training, including reproduction of the V1
  paper protocol. Canonical 75-epoch V1 runs with soft-reset NEST failed to fit
  excitatory firing rates. Matched early-training controls reproduce this
  suppression with both TensorFlow and fused NEST state updates, while legacy
  controls improve. Soft reset, passing gradient/replay tests and faster
  execution do not establish successful training or resolve this failure.

  Use NEST mode for inference or compatibility evaluation only after validating
  the intended network, reset mode, stimulus and population-level outputs.
  Legacy-trained weights can be evaluated in a separate NEST model, but a
  dynamics/reset change is not guaranteed to preserve trajectories or fitted
  observables. NEST training remains available for controlled method research;
  its presence in the API is not an endorsement for scientific training runs.

Set ``dynamics_mode="nest"`` explicitly to use NEST-compatible refractory timing,
time-averaged adaptation current,
spike-boundary adaptation reset, and exact alpha-current-to-voltage integration.
In NEST mode, SONATA initial voltage, adaptation state and recurrent/external
delays are honored. Times are converted through NEST's 0.001-ms ticks and then
rounded upward to simulation steps; ``dt`` must be a positive multiple of 0.001
ms and external delays must be at least one step. Reported spikes and voltage
samples use end-of-step timestamps.

For inference-only NEST execution, use these ``rnn_cell_params``:

.. code-block:: json

  {
    "rnn_cell_params": {
      "dynamics_mode": "nest",
      "hard_reset": true,
      "use_fused_cuda": "auto",
      "use_fused_state": false
    }
  }

For controlled research into experimental NEST training only (not a recommended
training recipe), set ``hard_reset=false`` or omit it. An RNN with configured
training, or built with ``rnn.build(training=True)``, resolves omitted or ``null`` reset to
soft reset before constructing the cell. Explicit ``hard_reset=true`` raises a
``ValueError`` in DPointNet training, including legacy-mode training. This check
also applies to direct ``TrainingEngine`` execution and checkpoint preparation.

.. code-block:: json

  {
    "rnn_cell_params": {
      "dynamics_mode": "nest",
      "hard_reset": false,
      "use_fused_cuda": true,
      "use_fused_state": false
    }
  }

The configuration above is a research-only example, not a validated NEST training
recipe. Build the CUDA operators before using it. Soft reset retains
the direct voltage-state gradient through spikes. Hard reset cuts that path at
spikes and during refractory clamping; the spike surrogate still supplies some
gradients but does not restore the lost path. This safeguard does not claim that
hard-reset training is mathematically impossible or that soft reset solves all
long-horizon credit-assignment problems.

Inference-only construction and direct ``GLIF3Cell`` construction retain the
mode-dependent default: hard reset in NEST and soft reset in legacy. Direct cell
users writing their own ``GradientTape`` loops must select soft reset explicitly;
an arbitrary external tape cannot be detected by the RNN training guard.

A prebuilt hard-reset model is rejected for training even if its config dictionary
is subsequently changed. Build a separate soft-reset training model and transfer
weights explicitly instead of mutating a traced graph. Conversely, inference and
validation on a trained model keep its soft reset. For hard-reset evaluation,
create a separate inference-only model with the learned weights and report the
reset setting. Switching reset modes changes forward dynamics, not merely
gradients; evaluate that mismatch before drawing scientific conclusions.

NEST mode uses ``ExplicitStateRNN`` to preserve scalar integer noise counters and
external delay history in symbolic Keras models and across chunks. Supply complete
initial state from ``cell.zero_state``; cached state without required delay history
is rejected. The wrapper supports explicit state, not Keras ``stateful=true``.
Keras 2 symbolic calls pack the sequence and initial states into one input list;
the wrapper unpacks them and uses the backend recurrent loop. Keras 3 retains its
native ``inner_loop``. Both paths preserve integer counters and floating state.

With ``return_voltage_sequences=false``, NEST compact cell output is a pair:
compute-dtype spikes and the FP32 per-sample voltage penalty. The loop stores
these separately, avoiding a large FP32 spike sequence merely to hold the penalty.
The extractor retains its two-sequence-output interface and complete state.
Full-voltage and legacy output formats are unchanged. Explicit ``unroll=true``
uses internal FP32 packing for Keras compatibility before returning the typed
pair; the memory reduction applies to normal looped execution.

NEST has a separate CUDA state forward/backward operator selected by
``use_fused_state=true``. Rebuild the CUDA operators before enabling it. It requires
four synaptic bases, either triangular or Gaussian surrogate, FP32 or FP16 compute with FP32
variables, and int8 or int16 refractory state. For both NEST and legacy state
dispatch, ``"auto"`` falls back to TensorFlow for unsupported dtype policies
(such as ``mixed_bfloat16`` or ``float64``), even if the CUDA library is loaded.
Explicit ``true`` rejects an incompatible policy during cell construction with
the compute and variable dtypes in the error. ``false`` remains the default and
retains TensorFlow state updates.
The legacy state kernel is never used for NEST. Fused current projection is independent.

``use_fused_nest_event_vjp=true`` optionally moves NEST's existing attached-reset
and attached-ASC event-adjoint arithmetic into the NEST backward kernel. It
defaults to ``false`` and requires ``dynamics_mode="nest"``, enabled fused state,
and a rebuilt library containing ``DpointnetNestStateBackwardEvents``. Explicit
unsupported requests raise an error rather than silently changing the derivative.
The old backward ABI and default path remain available. This changes neither
forward equations nor the reset/ASC attachment settings; it does not fuse spike
history or change replay, losses, canonical weights, or mixed-precision shadows.
The option supports ordinary per-state gradients and selective FP32 temporal
record/recompute execution. Separate dtype-rounded intermediate arithmetic is
retained, but bitwise gradient equality across TensorFlow/CUDA math implementations
is not guaranteed; qualify the intended precision and surrogate combination.
It is a backward-only option, not an inference accelerator or a general NEST
training recommendation.

The NEST kernel preserves alpha-current voltage integration, adaptation hold/reset,
hard/soft reset ordering, pre-reset spike surrogates, and delayed spike history.
Gradients cover floating state, currents and history; neuron coefficients remain
nontrainable. External delay history and Poisson counters remain in the cell wrapper.
Backward saves a compute-dtype 0/1 refractory mask to avoid CPU integer TensorList
transfers, restoring the integer dtype before the kernel. Forward refractory
counts and checkpoint state are unchanged. Coefficients are packed at execution
time so assignment and checkpoint restoration do not leave stale values.
The kernel rounds intermediate operations in compute dtype and uses the voltage-sum
association observed in optimized TensorFlow graphs. Floating-point results are not
guaranteed bitwise across eager/graph execution, optimizer settings or TensorFlow
versions; validate threshold-sensitive trajectories for the intended configuration.

Exact integration of a fitted alpha basis does not make that basis identical to the
source synapse model; validate synaptic approximation, precision and network-level
behavior separately. The historical timings below predate the NEST state kernel.

Performance qualification
~~~~~~~~~~~~~~~~~~~~~~~~~

The training-step measurements below qualify execution cost only. **NEST training
is experimental and is not recommended for use**, irrespective of these speed or
memory improvements. The convergence limitation above remains unresolved; this
implementation does not change the training recipe or silently substitute legacy
integration/gradients to make NEST training succeed.

The latest 2026-09-17 separate-output follow-up on the workload below measured
NEST at 4.55 s/update and 10.60 GiB timed peak, versus matched legacy at 4.36 s
and 10.61 GiB, using default TensorFlow memory optimization. This is about 4.4%
more time and effectively equal allocation. The private ``NO_MEM_OPT`` override
measured NEST at 4.59 s with the same peak and has no demonstrated benefit on
this graph. All use ``cuda_malloc_async``; default BFC was not retested. Three
warmups and 20 synchronized samples per run, matched source/binaries/inputs,
and no convergence or other-GPU guarantee. The earlier timings below used
combined FP32 compact outputs and are retained as historical measurements.

A 2026-09-17 optimized-kernel follow-up on the workload below measured NEST at
4.80 s/update versus a fresh matched legacy control at 4.41 s (8.7% slower), with
15.57 versus 10.61 GiB timed peak allocation. Both used a benchmark-only private
TensorFlow ``NO_MEM_OPT`` override, not a library default. With default TensorFlow
memory optimization, NEST measured 6.63 s and 17.12 GiB. Each run excluded three
warmups and timed 20 updates with matching source/binary provenance and inputs.
Saving a floating backward mask eliminated repeated integer TensorList transfers;
native rounded FP16 instructions alone did not demonstrate whole-update speedup.
NEST retains FP32 compact outputs and heterogeneous state, whereas ordinary Keras
RNN casts legacy outputs/state to compute dtype. The remaining memory gap is not
fully attributed. Do not treat diagnostic performance as ordinary-default speed
or silently reduce NEST penalty precision. These changes remain opt-in.

A 2026-09-16 RTX 3090 benchmark measured median training updates of 4.38 s for
legacy fused state, 5.93 s for legacy TensorFlow state, and 8.66 s for NEST
TensorFlow state. All retained fused CUDA currents. The workload used 66,658
neurons, FP16 compute, effective batch 32 (parallel 16+16), 500 timesteps, nine
paper losses, soft reset, exact chunk-25 BPTT, and compact online voltage loss.
Each run excluded three warmups and timed 20 synchronized updates with matched
inputs, source, and ``TF_GPU_ALLOCATOR=cuda_malloc_async`` under TensorFlow 2.21.

NEST took about twice the legacy fused-state time; disabling fused state alone
increased legacy time by 35%. The remaining difference also includes changed
activity and state/output handling. Default BFC allocation failed in NEST
backward on this 24 GiB GPU; the async allocator completed without reducing batch
size or changing losses. This is a workload-specific feasibility result, not a
universal allocator recommendation. These short soft-reset updates do not
establish convergence, hard-reset inference speed, or full-network NEST parity.
Choose dynamics for the scientific protocol and do not silently switch modes.

CUDA and fallback regression agreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/simulator/dpointnet/test_nest_dynamics.py`` compares fixed-input CPU,
GPU TensorFlow fallback and fused CUDA rollouts, including both reset modes,
float32/mixed float16, 1/0.25-ms steps, low/strong drive, gradients and chunk state.
A separate check compares seeded internal Poisson histories exactly. The bounded
fixture requires equal per-neuron spike counts, spike-time differences of at most
one step, and exact noise counters/external delay histories. When spike times are
identical, continuous states and gradients must satisfy rtol/atol 1e-5 (float32)
or 2e-2 (float16). If jitter occurs, unshifted numerical errors are reported in
JUnit properties but pointwise continuous-state/gradient parity is not asserted.
Existing individual kernel tolerances remain unchanged.

This is not bitwise equality: the tested lower-drive float16/0.25-ms cases can
shift a spike by one step, producing a large instantaneous voltage difference
at reset. All tested 1-ms cases matched spike times, but that is not a full-network
or cross-architecture guarantee. Run with ``--junitxml=REPORT.xml`` to retain
per-case timing, spike jitter, state errors, gradient errors and whether continuous
parity was checked. Setting ``CUDA_VISIBLE_DEVICES=''`` tests CPU-only support;
CUDA-specific comparisons then skip rather than falsely reporting a CUDA pass.

Optional fused CUDA operator
----------------------------

DPointNet can use a fused CUDA operator for recurrent and input synaptic currents. This optional operator
requires an NVIDIA CUDA toolkit with ``nvcc``, a C++17 compiler, and a GPU-enabled TensorFlow installation.
Build it from the same environment in which BMTK and TensorFlow are installed:

::

  $ python -m bmtk.simulator.dpointnet.custom_ops.build

This module form ensures that the build uses the active Python environment and does not require a console script on
``PATH``. Installing this version of BMTK also creates ``bmtk-build-dpointnet-cuda`` in the environment's executable
directory; the shortcut is available when that environment is activated.

The build targets compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, and 12.0 by default, with
``compute_120`` PTX. To build for a different set of architectures, provide space-separated architecture numbers:

::

  $ DPOINTNET_CUDA_ARCHS="80 86 90" python -m bmtk.simulator.dpointnet.custom_ops.build

The operator requires exactly one visible GPU. Set ``use_fused_cuda`` to ``true`` in ``rnn_cell_params`` to
require the operator, or to ``"auto"`` to use it when available and otherwise fall back to TensorFlow. The
default is ``false``. Rebuild the operator after changing TensorFlow or CUDA installations or after updating
BMTK custom-op source; SavedModel graphs containing an older custom-op signature must also be regenerated.

Recurrent backward-kernel selection is controlled separately by ``use_pair_projection``:

* ``"auto"`` (default) selects the pair-projected kernel when fused CUDA is active, the configured batch size
  is 32, and the synaptic basis has four columns. Other configurations use the general fused backward kernel.
* ``true`` explicitly enables pair projection for any positive configured batch size and basis width with
  fused CUDA. Shapes outside the batch-32 specialization use a general projection kernel and the general
  backward reduction. This opt-in path does not require SM86 or packed backward.
* ``false`` always uses the general fused backward kernel and avoids compact-pair metadata construction.

For example:

::

  "rnn_cell_params": {
    "use_fused_cuda": "auto",
    "use_pair_projection": "auto"
  }

The pair-projected kernel computes each distinct postsynaptic-neuron/synapse-type basis projection once and
reuses it across recurrent edges. It only changes the recurrent backward pass; inference and learning rules
that do not differentiate through the recurrent dynamics do not benefit. Master weights and checkpoints remain
in canonical edge order.

On SM86 or newer GPUs, ``use_packed_sm120_backward="auto"`` additionally selects a packed FP32 recurrent
backward for float16 batch-32 models with four basis columns and ``uint32`` compact-pair metadata. Set it to
``false`` for a same-GPU comparison with the prior pair-projected kernel, or to ``true`` to require the packed
path and fail when any prerequisite is absent. It defaults to ``"auto"``; SM80 and older retain the prior
pair-projected kernel. The option name is retained for configuration compatibility after qualification on SM86.
The default CUDA build includes native SM86, SM89, and SM120 code plus ``compute_120`` PTX.

In the derived temporal-sweetspot source, float32 recurrent backward with batch32,
four basis columns and ``uint32`` compact-pair metadata reuses this register-packed
reduction automatically, including direct-CSR gradient output. It keeps projections,
products, accumulation and returned spike/weight gradients FP32; no FP16 or TF32
operand narrowing is introduced. The float16-only ``use_packed_sm120_backward``
option is unchanged; ``false`` does not disable this float32 specialization.
Disable pair projection to exercise the general float32 reference path. Other
shapes/index types and weight-only named-input backward retain their previous dispatch.

Each two-warp block traverses one source row, loads four consecutive projected
batch values per lane, and writes each edge gradient once without atomics.
All-zero source rows still compute their spike adjoints. Positive nonbinary
counts/fractional spikes are multiplied, not treated as binary events. Existing
signed-input behavior is preserved: direct-CSR weight backward gates nonpositive
spikes, whereas the older canonical pair path uses the signed multiplier.
For nonnegative spike inputs both paths implement the same VJP.
Reduction order changes, so qualification uses an independent FP64 edge oracle
and FP32 rounding bounds, including empty/silent rows and high fanout.
This derived implementation has CPU lane-layout/oracle tests only until rebuilt
GPU tests and whole-update measurements qualify it; no speedup, architecture
qualification or full-network memory result is implied.

``use_packed_sm120_external_backward`` controls the corresponding weight-only backward for named input
populations. It has the same ``"auto"``, ``true``, and ``false`` selection contract and additionally builds
compact pair metadata only for trainable input weights. Fixed LGN connectivity therefore retains its smaller
metadata layout. The packed kernel splits sparse source rows across blocks, returns no external-activity
gradient, and writes weight gradients through the CSR-to-canonical edge map. Fixed input populations skip both
backward kernels. SM80 and older and incompatible shapes retain the existing external backward.

For batch 32 with four basis columns, fused recurrent and spike-input forward passes group each source row's
active batch samples into a 32-bit mask and launch one CUDA block per active source row. This avoids launching
one block for every batch/source combination when biological spike tensors are sparse; other shapes retain the
general forward kernel.

Set ``use_active_row_forward=true`` to opt into that source-row compaction for batch sizes 1 through 32
with four basis columns. The mask reads only actual samples; it does not pad the physical batch or discard
simulations. The default remains ``false`` (the existing automatic batch-32 behavior is unchanged).
Half-precision scatter addition order and rounding differ from the general forward path, so numerical
and task-level validation is required; bitwise spike-trajectory identity is not promised.

Set ``use_small_batch_recurrent_backward=true`` to opt into a batch-1-through-8 recurrent backward kernel.
One block traverses each presynaptic row, reduces sample contributions locally, and writes each FP32
weight gradient once instead of atomically combining separate sample blocks. It supports general basis
widths and can be combined independently with explicit pair projection and CSR gradient output. It is
disabled by default: correct small-tensor behavior does not guarantee better full-network performance.

Set ``use_direct_csr_recurrent_gradient=true`` to accumulate dynamics weight gradients in contiguous CSR
order, then restore the existing master-variable order once at the rollout boundary. This now supports
general fused backward kernels, float16/float32, and general batch/basis sizes; it requires individually
trainable recurrent edges but not packed backward or SM86. Segmented BPTT uses its existing final
gradient transform. Full BPTT uses a retained tape without recomputing the forward rollout. Only the
rollout's gradients are transformed: outer weight-regularizer gradients remain in master-variable order
and are combined after restoration. Optimizer variables, slots, and constraints retain their ordering.

These paths are independent opt-ins, not a change to batch composition, learning rate, loss definitions,
sequential/parallel update semantics, or stopping criteria. On a short tuned V1 batch-5 full-BPTT series
benchmark (RTX 3090, 500 steps, all nine paper losses), active-row forwarding plus pair projection
measured 5.40 s per two-update step versus 7.41--7.52 s baseline. Direct CSR alone was approximately
neutral (7.44 s) and used more memory; small-batch backward was slower (10.98 s, or 8.49 s with direct
CSR). These are short-run results, not full-training convergence or other-architecture qualification.

Optimizer gradient safeguards
-----------------------------

``training.optimizer`` accepts ``name`` (``adam``, ``exp_adam``, or ``sgd``) and
one optional clipping mode: ``clipnorm`` clips each gradient tensor's norm,
``clipvalue`` clips each element's absolute value, and ``global_clipnorm`` clips
the combined norm across gradient tensors in an optimizer update. Thresholds
must be finite positive numbers. Omit these fields or use ``null`` to disable
clipping; multiple non-null clipping modes are rejected on both Keras 2 and 3.

For example, ``"optimizer": {"name": "exp_adam", "epsilon": 1e-11,
"global_clipnorm": 5.0}`` enables global-norm clipping. This threshold is
illustrative, not a calibrated training recommendation. Learning rate remains
in the separate ``training.learning_rate`` configuration.

The factory also accepts ``epsilon`` for Adam/ExponentiatedAdam (default
``1e-11``), and ``momentum``/``nesterov`` for SGD (defaults ``0.0``/``false``).
Other fields raise an explicit error instead of being silently ignored. Pass an
already-configured Keras optimizer to the Python API for additional constructor
options, without supplying ``optimizer_params``. The training-config ``name``
must match the factory selector; it is not a Keras instance name.

Clipping is configured on the inner optimizer before ``LossScaleOptimizer``
wrapping, so it acts on unscaled gradients. Custom training loops must use
``scale_loss_for_optimizer`` and ``unscale_gradients_for_optimizer`` as the
DPointNet training engine does: Keras 2 explicitly unscales before applying
gradients; Keras 3 unscales inside the wrapper. Do not clip scaled gradients or
unscale Keras 3 gradients a second time. Clipping neither prevents overflow
inside backpropagation nor bounds the final adaptive/multiplicative weight
update. Default no-clipping behavior is unchanged.

Activating the measured batch-5 speedup
-------------------------------------

Merge the following settings into an existing paper training configuration, retaining its inputs,
losses, initialization and callbacks. This is an opt-in execution profile, not a new learning-rate
or stopping policy. At batch 5, ``use_pair_projection="auto"`` does **not** enable projection;
use the JSON boolean ``true`` explicitly, together with ``use_active_row_forward=true``.

.. code-block:: json

  {
    "run": {
      "batch_size": 5,
      "seq_len": 500,
      "dt": 1.0,
      "dtype": "float16"
    },
    "rnn_cell_params": {
      "dynamics_mode": "legacy",
      "use_fused_cuda": true,
      "use_fused_state": true,
      "use_active_row_forward": true,
      "use_pair_projection": true,
      "use_packed_sm120_backward": false,
      "use_packed_sm120_external_backward": false,
      "use_small_batch_recurrent_backward": false,
      "use_direct_csr_recurrent_gradient": false,
      "use_fixed4_input_forward": false,
      "use_fused_current_accumulation": false,
      "track_voltage_penalty": false,
      "return_voltage_sequences": true
    },
    "training": {
      "training_approach": "series",
      "n_epochs": 75,
      "steps_per_epoch": 25,
      "gradient_checkpointing": false,
      "pack_spike_checkpoints": false,
      "optimizer": {"name": "exp_adam", "epsilon": 1e-11},
      "learning_rate": {"schedule": "none", "learning_rate": 0.005}
    }
  }

If individual training parameters specify ``batch_size``, keep each at 5 as well. With evoked
and spontaneous parameters, series mode still performs two sequential optimizer updates per
outer step: 3,750 updates over 75 x 25 steps. There is no batch padding or joint-condition update.
Keep offline voltage loss in both conditions (``online=false`` or omit ``online``).

Rebuild the CUDA operators from the updated checkout in the same environment used to train.
For the measured RTX 3090 configuration:

::

  $ DPOINTNET_CUDA_ARCHS="86" python -m bmtk.simulator.dpointnet.custom_ops.build
  $ python -c "import bmtk; print(bmtk.__file__)"

Choose the architecture for the actual GPU, and verify that the printed import path is the intended
checkout. Existing installations do not pick up an uninstalled source checkout automatically.
The packed-kernel flags above are disabled because batch 5 cannot use those specializations.
General direct CSR and the experimental small-batch reduction are also disabled because they did
not improve this full-network benchmark. Library defaults remain unchanged; an unmodified paper
configuration will not automatically enable the two new opt-ins. A 27--28% reduction in the measured
training-step time is not a measured 27--28% reduction in complete 75-epoch job time. Validate the
long-run behavior and hardware of interest before adopting this profile for production.

SONATA weight export restores the source edge-row order after runtime recurrent sorting. Interleaved
edge groups use ``edge_group_index`` to align properties with population rows, and multiple edge
populations retain separate input offsets. Physical weight export factors are applied before restoring
row order. The exporter still writes its existing single weight group; this is not a byte-for-byte copy
of every input HDF5 attribute, group structure, or index dataset.

Set ``use_fixed4_input_forward=true`` to select a fixed-four gather forward for input populations with exactly
four incoming edges per postsynaptic neuron. One thread owns each ``(batch, post)`` output and writes all four
basis values without scatter atomics. Selection is derived from connectivity structure rather than population
name; nonqualifying populations retain grouped/general forwarding. The default is ``false`` to preserve prior
mixed-precision summation semantics. Backward continues to use the source-CSR path, preserving canonical
trainable-weight gradients and the no-activity-gradient contract.

Set ``use_fused_current_accumulation=true`` to thread recurrent and fused spike-input currents through one
additive CUDA buffer instead of materializing each source and combining them with a separate ``AddN``. The
option defaults to ``false`` and requires fused CUDA currents. Its custom gradient passes the upstream current
gradient unchanged through the accumulator while retaining independent canonical gradients for every trainable
weight surface. Current-type or otherwise non-fused inputs retain the existing TensorFlow addition path.

The optimization exchanges startup time and a small amount of persistent GPU memory for faster batch-32 BPTT.
Measured examples include a 55.7% update-time reduction on a 66,658-neuron network on A100-PCIE-40GB and a 37.9%
reduction on a 19,570-neuron network on RTX 3090. The corresponding peak-memory increases were 0.54% and 0.09%.
These measurements are hardware- and topology-dependent; benchmark representative training before forcing the
pair kernel. Small-batch specializations were tested but regressed the complete networks, so ``"auto"`` does not
select pair projection below batch 32.

The optional ``use_fused_state`` cell parameter fuses the GLIF membrane,
refractory, ASC, PSC, spike, and delayed-history transition. It is ``false`` by
default. ``"auto"`` selects it only when the CUDA library is available, the
synaptic basis has four columns, and the dtype policy is supported;
``true`` requires those conditions and otherwise raises a configuration error.
Both surrogate shapes support fused execution. Soft reset avoids
retaining refractory history in the custom backward; hard reset remains
supported and tested.

Selective state precision and event credit
-----------------------------------------

The opt-in ``rnn_cell_params.state_precision="selective"`` policy requires
``mixed_float16`` (FP16 projection computation with FP32 master variables).
The default ``"compute"`` retains homogeneous floating state, historical legacy
ASC rate/logit/sigmoid rounding, and FP16-rounded NEST arithmetic. It does not
silently adopt the new policy.

Selective precision stores voltage and both ASC components in FP32, with all
derived neuron/synapse coefficients computed from float64 host inputs before
their final FP32 storage. Legacy selective ASC decay bypasses the historical
logit round trip. Synaptic rise and PSC blocks remain FP16; their update
arithmetic and voltage/ASC updates are FP32 before narrowing those two blocks.
For four synaptic bases this is **28 bytes per neuron per sample** for voltage,
ASC, rise and PSC, not total training memory. Spike/delay history, refractory
counters, coefficients, checkpoints, gradients and optimizer state are extra.
With the default ``temporal_gradient_precision="compute"``, stored-state adjoints
and the projected-current gradient still cross FP16 boundaries: FP32 voltage/ASC
alone does not guarantee long-horizon gradient
accuracy or prevent underflow. This is an experimental numerical policy, not a
qualified convergence or performance recommendation.

Both legacy and NEST use the explicit-state RNN for this heterogeneous policy.
Compact output preserves separate FP16 spikes and FP32 voltage penalties;
full-voltage output packs spikes with FP32 voltages and therefore uses FP32.
Exact segmented recomputation and explicit chunk continuation retain each
state tensor's dtype; no common-dtype state concatenation is used. Loading
canonical FP32 weights is independent of the state policy. To continue an old
compute-policy state, explicitly cast its voltage and ASC tensors to FP32
(and its synaptic/history tensors to FP16 if necessary); selective RNNs reject
unconverted voltage/ASC state. Cross-policy trajectories need not be identical.
Keep ``state_precision`` and all gradient settings in the saved JSON model
configuration; a state-only checkpoint does not encode their meaning.

``detach_reset`` and ``detach_asc_reset`` independently control spike-event
credit and default to ``true`` for backward compatibility. They leave forward
thresholds, reset maps, refractory counters and spike histories unchanged:

* Legacy attachment supplies reset sensitivity ``-1`` (unless the hard-reset
  refractory clamp blocks voltage) and adaptation sensitivity ``A``.
* NEST adaptation attachment supplies ``A + (rho - 1) * a_minus``, including
  both signed components and the refractory event multiplier. Continuous ASC
  state derivatives are preserved whether attachment is enabled or not.
* NEST reset attachment supplies ``-(1 - V_reset)`` for soft reset or
  ``V_reset - V_before_reset`` for hard-reset evaluation. NEST refractory
  gating still suppresses the event surrogate. Training continues to reject
  hard reset; these flags do not widen that API.

``pseudo_gauss=true`` selects ``amplitude * exp(-u**2 / gauss_std**2)`` for
normalized threshold distance ``u``. ``dampening_factor`` is the nonnegative
finite amplitude, and ``gauss_std`` must be finite and positive; it is the
width of this expression, not the standard deviation of a normalized Gaussian
density. The triangular alternative remains
``amplitude * max(1 - abs(u), 0)``. Both keep the same hard forward threshold,
including its strict ``u > 0`` decision and refractory mask. Gaussian symmetry
is around voltage, not time: no future inputs enter the forward transition.
BPTT may assign retrospective credit from later losses, which is not an
online or strict-local learning rule.

Rebuild ``python -m bmtk.simulator.dpointnet.custom_ops.build`` after updating.
The state CUDA ABI now separates voltage/coefficient type ``T`` from
synaptic/history type ``S`` and includes the Gaussian backward V2 symbol.
Stale libraries cannot satisfy explicit fused-state selection. Canonical
weight masters, projection acceleration, compute-shadow refresh, Poisson
streams and exact-checkpoint boundary semantics are unchanged.
The NEST wrapper explicitly stops autodiff through the opaque CUDA forward op
inside its custom VJP, allowing nested full-BPTT/direct-CSR tapes to use the
supplied state gradients. This does not stop gradients across time or checkpoints.
Both full and segmented accelerated updates require qualification on the actual
topology; small-fixture agreement is not full-network memory or speed evidence.

FP32 temporal cotangents with quantized forward state
----------------------------------------------------

Selective forward precision alone does not ensure long-credit accuracy. Repeated
FP16 adjoint rounding can retain a nonzero subnormal fixed point, rather than only
underflowing to zero. Loss scaling reduces this floor but does not generally
remove it. The opt-in ``temporal_gradient_precision="float32"`` policy addresses
that temporal boundary structurally; the default ``"compute"`` preserves the
original backward policy.

Only actual external continuous-current tensor surfaces require FP32 inputs.
Input ``options.input_type`` takes precedence over the population's nominal
``input_type``: internally sampled Poisson background populations do not consume
an external continuous-current tensor. LGN-only external spike inputs may remain
boolean, including when combined with internal Poisson background. Genuine
external continuous inputs retain the FP32 public-boundary guard; they must not
be narrowed by the LGN boolean-input optimization.

This option requires ``state_precision="selective"`` and mixed_float16. The
forward rollout and saved checkpoints still use the **28-byte** voltage/ASC/PSC/
rise layout. The custom reverse loop replays each chunk with FP32 shadow values
which remain exactly quantized to the forward FP16 storage values, but have an
FP32 storage Jacobian. Cotangents remain FP32 at every replay timestep and across
every chunk boundary. Current projection uses the actual quantized forward value
with its FP32 linear VJP at the quantized weights and basis values; it does not
substitute an unquantized FP32 forward trajectory.

Exact replay must also be qualified on the actual connectivity and weights.
The grouped FP16 CSR forward uses floating-point atomic accumulation, so repeated
projections can differ for non-dyadic weights. The native FP32-adjoint runner resolves an omitted/null
``rnn_cell_params.current_replay_mode`` to ``"record"`` and records each original
timestep's summed projected currents **after**
recurrent/named-input accumulation and the cell's ``lr_scale``. It retains one
FP16 time-major tensor per chunk explicitly on CPU, not one tape per source and
not a second full GPU stack. Reconstruction and FP32 shadow replay reuse these
original values, while the existing FP32 projection VJP still differentiates
the canonical weights and source spikes. Quantized projection weights and basis
values are also snapshotted per rollout, preserving gradients if weights change
before reverse execution. RNG snapshots and current tapes are invocation-local;
multiple forwards/backwards cannot overwrite another rollout's saved values.

``current_replay_mode="recompute"`` is an explicit **approximate replay** option.
It creates no original-current tape, captures no timestep currents, and instead
reruns the original compute-dtype (FP16) projection and accumulation during
reverse execution. It uses the invocation's saved quantized weights and basis,
not live mutable weight shadows, and does not replace FP16 projection with an
FP32 projection followed by a cast. The forward-time logical RNG seed and
checkpoint noise-step/external-delay history preserve Poisson counts, including
counts greater than one. Both modes snapshot weights, basis and RNG even when
multiple forwards precede backward or weights are mutated before backward.
Replay never temporarily assigns the saved weights into shared shadows.

This switch does not change ``state_precision``, ``temporal_gradient_precision``,
FP32 voltage/ASC/cotangents, projection precision, accelerator settings, Gaussian
surrogates or reset flags. It applies only to the native FP32 temporal runner.
The constructor default is ``None`` (JSON ``null``), not ``"record"``.
``cell.current_replay_mode_requested`` retains the requested value;
``cell.current_replay_mode`` reports the resolved value, with ``None`` meaning
**inactive**, not recording. The supported matrix is:

.. list-table::
   :header-rows: 1
   :widths: 22 25 53

   * - Temporal gradient precision
     - Requested current replay mode
     - Resolution
   * - ``"compute"``
     - Omitted / ``None`` / JSON ``null``
     - Inactive (``None``). Existing ordinary BPTT or generic checkpointing is unchanged; no original-current tape is enabled.
   * - ``"compute"``
     - ``"record"`` or ``"recompute"``
     - Error. Neither explicit mode is supported on this route.
   * - ``"float32"``
     - Omitted / ``None`` / JSON ``null``
     - ``"record"``: original-current CPU tape, preserving the existing FP32-carry default.
   * - ``"float32"``
     - ``"record"``
     - Original-current CPU tape.
   * - ``"float32"``
     - ``"recompute"``
     - No current tape; approximate FP16 reprojection with a warning.
   * - Either valid precision
     - Any other value
     - Error.

The matrix applies to both legacy and NEST dynamics. Existing precision
requirements still apply: FP32 temporal carry requires ``state_precision="selective"``
with mixed_float16; ordinary compute carry retains its existing compute/selective
state policies. No FP16-carry current-tape mode is implemented. Existing normal
configs that omit the new option keep their prior behavior without advertising
an active tape. Configure the option in ``rnn_cell_params`` or the ``GLIF3Cell``
constructor and retain it in the saved model JSON; the RNN factory forwards it
unchanged. No independent runner override is provided.

Atomic FP16 projection is not bitwise deterministic. Recompute therefore emits
a warning and may replay different currents, states, spikes and gradients even
at the saved weights/RNG. Exact agreement on deterministic CPU fixtures does not
qualify atomic GPU replay. Deliberately changing-projection tests require this
mismatch for recompute while retaining the strict original-current gate for
record. Never weaken that record gate to accommodate recompute.

The accelerated tape is a reference-counted CPU tensor resource. Its loop-carried
handle does not copy previously recorded chunks across devices, and reads/writes
share CPU tensor-buffer ownership rather than copying values elementwise.
Rebuild the custom operators to provide ``DpointnetCurrentTapeCreate``,
``DpointnetCurrentTapeWrite`` and ``DpointnetCurrentTapeRead``. CUDA-disabled
execution uses an anonymous integer table with lossless FP16 bit packing for
compatibility; it is not the performance path. Output-gradient conversion to
FP32 occurs after slicing the current reverse chunk, avoiding a full-sequence
FP32 gradient allocation. Neither change narrows the carried temporal adjoints.

Fused FP32 replay uses the internal ``vjp_only`` sparse-op mode to avoid
recomputing an unused FP32 projected primal. This mode produces a zero placeholder
solely under quantized-primal substitution (recorded or freshly reprojected
FP16 currents) and retains the registered sparse VJP. Ordinary forward projection
defaults to ``vjp_only=False``. Rebuild the operators when adopting this ABI
attribute; do not use the placeholder as a forward simulation result.

The current tape costs ``batch * neurons * timesteps * bases * 2`` host bytes:
batch32,66,658 neurons,500 timesteps and4 bases require7.946GiB host memory.
Only one chunk (0.397GiB at chunk25) is transferred back for reverse execution.
Recompute avoids this host-current storage/transfer, but adds FP16 projection
work during reverse and can change the approximate gradient trajectory.
Neither a speedup nor a five-second update is promised. GPU memory, timing,
gradient discrepancy and actual-network qualification remain required for each
mode; CPU correctness and Keras compatibility do not establish those results.
This excludes checkpoint states, spikes, weights, activations and gradients.
Device memory and transfer overhead still require actual-network qualification.
Persistent epoch checkpoints store model/optimizer/RNG state, not an in-flight
reverse tape. A new original forward can still differ due to atomic ordering;
bitwise replay guarantees apply to its recorded original trajectory, not to two
independently projected forward calls. Never relax the actual-network replay gate.

Current-replay qualification plan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python-only replay-mode switch does not change the custom-op ABI. Existing
compatible optimized operators remain required for GPU execution. Its initial
qualification is CPU-only on TensorFlow2.21/Python3.13 and actual
TensorFlow2.13.1/Python3.8 (Keras2), using CUDA-disabled execution of:

* ``tests/simulator/dpointnet/test_current_replay_mode.py``
* ``tests/simulator/dpointnet/test_temporal_adjoint.py``
* ``tests/simulator/dpointnet/test_temporal_current_tape.py``
* ``tests/simulator/dpointnet/test_temporal_host_tape.py``
* ``tests/simulator/dpointnet/test_precision_credit.py``

These cover both dynamics/policies, partial chunks, independent analytic
600-ms adjoints, FP32 carried cotangents, absence of a recompute host tape,
FP16-versus-FP32 projection discrimination, JSON/RNN routing, invocation-local
RNG/weights/basis, graph-mode multiple-forward mutation, and deliberately
non-repeatable projections. CUDA-specific tests are skipped, not qualified.

Before use on a real GPU workload, separately qualify unchanged record replay
and quantify recompute current/state/spike/gradient discrepancies at matched
weights and logical RNG. Preserve precision, input delays/counts, dynamics,
reset/surrogate flags and all selected accelerators; reject unsupported explicit
requests instead of silently falling back. Run a small forward/backward and
optimizer smoke before full-network comparisons. Report default-BFC device and
host memory plus synchronized whole-update timing with at least three excluded
warmups and twenty samples. Record remains the exact-replay control; recompute
is an approximation/performance tradeoff, never an automatic replacement.
GPU execution and benchmark claims are pending coordinator qualification.

Private replay-cache contract for diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TemporalAdjointRunner._forward(inputs, initial_state, probe_steps=())`` retains
the existing seven-field tuple, in this exact order:

``(sequences, final_state, boundaries, saved, seed, tape, projection_values)``.

* ``sequences`` is always a tuple of batch-major full-sequence outputs. Its
  components follow ``cell.output_size`` (compact spikes/penalty, or the single
  combined spike/voltage output). ``final_state`` uses the original storage dtypes.
* ``boundaries`` is the sorted, unique timestep tensor, including zero, the final
  length, regular chunk boundaries and requested probes. Iterate
  ``tf.size(boundaries)-1``, not ``ceil(length/chunk_size)``.
* ``saved`` is a tuple of TensorArrays indexed by boundary, including initial and
  final states. Spike history alone may be packed int32; unpack it using the
  initial spike-history width/dtype before replay or comparison. These are
  **boundary states**, not every timestep's voltage/ASC/PSC trajectory.
* ``seed`` is the invocation-local logical RNG seed. Noise step and NEST input
  delay history remain in the saved recurrent state.
* ``tape`` is the original CPU current resource in record mode and ``None`` in
  recompute. Recorded chunks are time-major ``[chunk_time, batch, neurons*bases]``.
* ``projection_values`` is ``(quantized_basis, tuple(quantized_weights))``.
  Weights are recurrent first, then inputs in insertion order. Discrete CUDA
  projections use CSR shadows; TensorFlow projections use canonical shadows;
  continuous-current weights are quantized canonical masters. Both modes retain
  these snapshots. Do not replace them with live weights.

Use ``runner._replay_cached_chunk(inputs, initial_state, cache, index)`` for a
diagnostic chunk. It returns
``(replay_outputs, replay_state, original_outputs, original_boundary_state)``.
It handles packed history, partial/probe chunks, snapshot forwarding and the
mode's tape selection. It rejects inconsistent caches/manual
``record_currents`` toggles. This is a private diagnostic accessor, not an
alternate native training or benchmark path.

Equivalent direct ``_loop`` usage must promote floating checkpoint state to FP32
and supply **both** ``projection_context`` (prepared from the snapshot for FP32
VJPs) and ``projection_values`` (the same snapshot for original FP16 projection):

.. code-block:: python

   outputs, state = runner._loop(
       tf.cast(inputs[:, start:stop], tf.float32),
       replay_state32,  # unpacked original state at this chunk boundary
       replay=True,
       projection_context=cell._prepare_adjoint_projection_context(
           saved_values=projection_values),
       projection_values=projection_values,
       noise_seed=seed,
       recorded_currents=runner._read_current_chunk(tape, index),
   )

The context alone is insufficient for recompute. Do not set
``capture_currents=True`` in a recompute diagnostic, or mutate
``runner.record_currents`` to select modes. Construct separate FP32-carry
cells/runners using the public ``current_replay_mode`` option.

The tested ``cache_replay_error_metrics`` helper in
``tests/simulator/dpointnet/test_current_replay_mode.py`` is a copyable harness
example. It reduces each chunk immediately into int64 mismatch counts and
float64 maximum absolute errors, checks nonfinite values explicitly, reports
full-sequence spike/output discrepancies and **state-boundary** discrepancies
separately, and supports graph execution. Each replay chunk starts at its
original checkpoint, matching backward; this is not a free-running replay.
No whole-network replay-state stack is needed. Recompute has no original-current
tape, so this diagnostic cannot report original-versus-recomputed current errors.

Keep ``_forward`` and diagnostics in the same eager or traced scope; do not return
the Python resource/cache object across a ``tf.function`` boundary. Return only
the reduced metric tensors. Original cache collection and replay diagnostics
are excluded from native timings. Benchmark the real compiled training update
including losses, backward, optimizer and shadow refresh, with completion
synchronized after each call. Retain raw twenty-sample timings after three
excluded warmups; compare matched initial weights/RNG/input/loss/settings and
report any evolving-trajectory differences. Diagnostics do not replace this
native update measurement, and neither mode's result is a five-second promise.

The existing FP32 CUDA state and projection kernels execute these replay VJPs.
FP32 NEST state calls use a structure-of-arrays coefficient layout for coalesced
adjacent-neuron loads, with unchanged arithmetic and neuron/state ordering.
The raw forward/backward operators accept ``coefficients_layout="soa"`` for
``[28, neurons]`` FP32 coefficients; their default ``"aos"`` retains the
``[neurons, 28]`` layout and existing FP16 path. The wrapper prepares the
transpose from live coefficients and uses the same layout in both directions.
Rebuild operators before adopting this attribute.
Fused FP16 forward projection, current accumulation and canonical master/shadow
ordering remain intact. Eligible FP32 batch32/four-basis/uint32 compact-pair
recurrent gradients use a dedicated register-packed FP32 specialization, with
generic fallback for other shapes. This is separate from the FP16 packed
configuration flags. Explicit
``use_packed_sm120_backward=true``,
``use_packed_sm120_external_backward=true`` and
``use_small_batch_recurrent_backward=true`` are rejected with this policy.
Automatic FP16 packed selection does not apply to replay. Direct-CSR recurrent
gradients are restored to canonical order once by the temporal runner.

Backward memory is additional: the four primary replay-state blocks occupy
**44 bytes** per neuron/sample, not28. FP32 spike/delay cotangents, replay
activations and temporary FP32 quantized projection-weight copies are extra.
Projection copies are prepared outside the inner replay timestep loop.

Opt-in recurrent FP32 accumulation fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable the optional recurrent accumulation producer with
``rnn_cell_params.use_fused_recurrent_accumulation=true``. The default is
``false``. It requires FP32 temporal carry with selective mixed_float16 state,
batch32, four bases, trainable per-edge recurrent weights
(``train_recurrent_per_type=false``), direct-CSR recurrent gradients, uint32
compact-pair metadata, and rebuilt SM86+ CUDA operators. Unsupported explicit
requests fail; ordinary compute carry and generic/noneligible paths are
unchanged with the default. Both record and recompute current policies are
supported. Neuron-state fusion is independent of this option.

During reverse-chunk replay, an internal differentiable identity weight carrier
threads through the existing TensorFlow timestep loop. Its cotangent is the
running recurrent CSR gradient. The new ``DpointnetCsrSpikeGradAccumulate``
receives that cotangent as a real tensor input and adds the current contribution
in the existing packed recurrent producer. No full per-step gradient vector is
materialized for a subsequent TensorFlow add. The public cell state and the
seven-field forward cache remain unchanged.

The packed per-step reduction, FP32 products and addition precision are
preserved. The producer uses a round-to-nearest FP32 addition to the incoming
accumulator. Reverse timestep order and the existing separate chunk sums are
retained; chunk totals are still added outside the inner loop. Exactly one CSR
to canonical restoration occurs at the rollout boundary, before ordinary
regularizer-gradient composition. This is the same first-order BPTT
computation, not truncation, clipping, reduced precision or a new objective.

The forward carrier is an immutable identity. The producer uses TensorFlow's
``forward_input_or_allocate_output`` ownership check, allocating a new output
when forwarding is unsafe. It never assigns a global/Variable accumulator or
forces input aliasing. Retained accumulator inputs, branched/chained uses,
nonzero seeds, repeated backwards and weight/RNG mutation are tested. Buffer
forwarding frequency is not assumed or directly measured; memory claims must
come from actual peak measurements.

For24,450,554 edges, one FP32 vector is97,802,216 bytes. The verified target is
the500 per-step recurrent AddN operations, not the entire large Eigen kernel
group: other regularizer/state additions remain. A baseline trace attributed
about0.175s to this recurrent add and0.938s to recurrent gradient production.
Fusion removed the large per-step add and measured about1.007s in the combined
producer. The approximately20 outer chunk additions remain.

An identical recorded full-network forward comparison, including all configured
losses and direct regularizer gradients, found exactly equal recurrent gradients
and recurrent weights after a same-class optimizer update. Unchanged BKG atomic
reductions differed by at most3.73e-9 in gradients and5.37e-7 in updated weights.
The optimizer comparison uses cloned canonical weights/slots/constraints and
unscaled gradients; the real distributed binding and loss scaler are exercised
by separate native-update tests. Recompute forward nondeterminism is excluded
from this identical-cache comparison.

Actual Keras2 CPU tests cover the loop/carrier graph contract through a
TensorFlow test double and preserve ordinary RNN routes; the CUDA library here
is built for the current TensorFlow environment, not Keras2. Legacy fused-state
tracing of two persistent-tape backwards already fails in the unfused baseline
because a state-backward op has no higher-order registration. Persistent public
tape tests therefore use TensorFlow neuron state for legacy; repeated cached
VJPs also cover legacy fused state. This fusion does not add higher-order
derivative support.

The qualified plain-native timing boundary includes the compiled training step,
all losses/backward/optimizer work and gradient instrumentation, with explicit
synchronization. Restoration, hashing, replay diagnostics and separately timed
shadow-refresh checks are excluded. Three warmups and twenty samples were used
per fresh process. Initial medians were6.00880s unfused and5.87843/5.89177s fused.
A final same-source/library control pair measured5.99377s unfused and5.89644s
fused (about97ms or1.62% faster). This is a modest profile-specific improvement,
not a five-second result or a reason to change defaults universally.

There is **no demonstrated peak-memory saving**. With stats reset immediately
before each post-warmup compiled step and sampled after synchronization, median
current/peak allocation was4.291/14.960GiB unfused and4.409/15.082GiB fused.
Maximum per-step peaks were15.096/15.164GiB. Driver reservation/free memory was
20,807/3,451MiB in both modes. These matched per-update measurements exclude
setup/tracing; earlier process-lifetime peaks are reported separately.

The memory-instrumented timing pair was5.88846/5.89016s and did not show a
speedup. The added synchronization/stat sampling changes that measurement
boundary; this sensitivity and all raw samples are retained. Keep the feature
off by default and enable it only through explicit profile qualification.
The observed plain-native gain does not imply a memory benefit or portability
to other hardware, TensorFlow versions, losses or execution boundaries.
``temporal_checkpoint_chunk_size`` defaults to25 and bounds replay chunks; use a
size at least the sequence length for single-chunk/full replay. Optional
``temporal_pack_spike_checkpoints=true`` packs only binary spike-history
checkpoints. Diagnostic cotangent capture also retains boundary cotangents.
Neither28 nor44 bytes is a total training-memory estimate.

The normal ``RNN`` factory and ``ExplicitStateRNN`` select this reverse path
automatically. Training checkpoint settings are wired into the cell before model
construction; conflicting chunk sizes fail explicitly. Do not wrap this RNN in
``SegmentedRecomputeRunner`` or ``FullBPTTGradientRunner``: those outer wrappers
would narrow boundary gradients or restore CSR order twice, and are rejected.
Masked, time-major, backwards, stateful and explicitly unrolled RNN execution
are not supported by this new policy. The compatibility policy is unchanged.

For direct use:

.. code:: python

   from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner

   # Construct GLIF3Cell with state_precision="selective" and
   # temporal_gradient_precision="float32" under mixed_float16.
   runner = TemporalAdjointRunner(cell, chunk_size=25, pack_spike_checkpoints=True)
   initial = cell.zero_state(batch_size, cell.compute_dtype)
   outputs, final_state = runner(inputs_float32, initial)

   result = runner.differentiate(
       inputs_float32, initial,
       lambda outputs, state: voltage_objective(outputs),
       probe_steps=(100, 350, 500),
   )

``runner`` is differentiable with respect to trainable masters and continuous
FP32 inputs. ``differentiate`` returns ``loss``, ``outputs``, ``final_state``,
``input_gradients``, ``variable_gradients``, ``initial_state_gradients``,
``state_cotangents`` and ``floating_state_indices``. Each captured cotangent has
shape ``[probe, batch, state_width]`` in FP32. Probe steps denote the state
timestamp before the indexed step; they are added to checkpoint boundaries.
The supplied loss function receives FP32 copies of floating outputs.

Continuous-input populations must receive FP32 tensors. Their forward projection
still quantizes as prescribed by the compute policy, while the input VJP remains
FP32. A standard TensorFlow gradient returned to an explicitly FP16 initial-state
tensor is narrowed once at that external API boundary; use ``differentiate`` to
inspect the FP32 internal cotangents. Compact forward spikes remain FP16, so
loss scaling can still be needed for an outer loss connected to FP16 outputs.
This policy removes repeated temporal narrowing; it does not make every external
tensor interface FP32.

Direct ``cell.call`` or ``cell(...)`` with this option raises an explicit error:
a direct-cell tape cannot provide these temporal adjoints. Use the runner or RNN
wrapper instead. Exact replay preserves integer refractory/RNG state and
stateless Poisson samples without advancing the logical stream twice.
The base noise seed is snapshotted per rollout, so advancing the stream for a
second forward pass before differentiating the first cannot change its replay.
As with exact checkpointed BPTT, weights and neuron coefficients must not be
mutated between a rollout and its reverse pass.
This is first-order BPTT, not a qualification of higher-order derivatives or
100--500ms learning/convergence. In particular, a strongly driven synthetic
gradient fixture is not a physiological firing-rate experiment.


Overview
========

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
      

  .. tab-item:: Python

    .. code:: python





DPointNet Initialization
========================

Setting up a simulation environment/workspace
---------------------------------------------

Initializing DPointNet instance
-------------------------------

First we must instantiate a DPointNet simulator instance, which passing in parameters 
required to build and run the model. Some of these parameters may be changed later on.

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json

        {
          "target_simulator": "DPointNet",

          "run": {
            "seq_len": 500,
            "dt": 1.0,
            "default_seed": 3000,
            "batch_size": 10,
            "train_recurrent_weights": true,
            "dtype": "float32",
            "single_gpu_strategy": "one_device"
          }
        }
      

  .. tab-item:: Python

    .. code:: python

        from bmtk.simulator import dpointnet

        rnn = dpointnet.RNN(
            seqlen=500.0, 
            dt=1.0,
            default_seed=3000,
            batch_size=10,
            dtype="float32",
            train_recurrent_weights=True,
            single_gpu_strategy="one_device",
        )


.. dropdown:: Available "run" options
  :open:

    .. list-table::
        :header-rows: 1

        * - option
          - description
          - default
        * - seq_len
          - The number of time steps that will be used in training and inference 
          - 
        * - dt
          - The time interval, in milliseconds, for each sequence step
          - 1.0
        * - default_seed
          - The default RNG seed to use when building the model and any of DPointNet functions (like spike generators or training) - when not explicity stated.
          - 
        * - batch_size
          - The default number of simulataneous batches for processing during feed-forward input into the RNN. May be overridden for training and inference.
          - 1



Setting hyper-parameters for training/inference
-----------------------------------------------

Next we must set hyper-parameters that are used by the back-end deep-learning model. You must first specify the `cell_model` which takes care of reproducing
the internal simulation output plus rules for determining gradient calculations. Current DPointNet only supports `GLIF3Cell` model that reproduces 
the `GLIF point-neuron models <https://brain-map.org/our-research/computational-modeling/glif-single-neuron-models>`_.

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json

        {
          "rnn_cell_params": {
            "cell_model": "GLIF3Cell",
            "<parameter_1>": <value_1>,
            "<parameter_2>": <value_2>,
            "<parameter_3>": <value_3>,
            ...
          }
        }
      

  .. tab-item:: Python

    .. code:: python


.. dropdown:: Available "rnn_cell_params" options

    .. tab-set::

        .. tab-item:: GLIF3Cell

            .. list-table::
                :header-rows: 1

                * - option
                  - description
                  - default
                * - gauss_std
                  - Positive finite Gaussian surrogate width (not a normalized density standard deviation).
                  - 0.5
                * - dampening_factor
                  - Scale applied to the spike surrogate derivative.
                  - 0.3
                * - recurrent_dampening_factor
                  - Retained recurrent temporal-gradient multiplier. ``0.0`` blocks this gradient and ``1.0`` leaves it undampened.
                  - 0.5
                * - voltage_gradient_dampening
                  - Fraction removed from the membrane-voltage self-loop gradient: ``0`` retains it and ``1`` blocks it. Synaptic-current gradients are not scaled.
                  - 0.5
                * - detach_reset
                  - Stop only the spike-to-voltage-reset gradient. Forward reset is unchanged.
                  - True
                * - detach_asc_reset
                  - Stop only the spike-to-ASC-event gradient. Continuous ASC gradients are unchanged.
                  - True
                * - state_precision
                  - ``"compute"`` preserves homogeneous floating state; opt-in ``"selective"`` uses FP32 voltage/ASC/coefficients and FP16 synaptic storage under mixed_float16.
                  - "compute"
                * - temporal_gradient_precision
                  - ``"float32"`` selects the custom FP32 temporal reverse/replay path with selective forward storage. Direct cell tapes are unsupported.
                  - "compute"
                * - temporal_checkpoint_chunk_size
                  - Maximum replay chunk length for the FP32 temporal policy.
                  - 25
                * - use_fused_recurrent_accumulation
                  - Opt-in producer/temporal-accumulator fusion for trainable per-edge recurrent weights with FP32 temporal carry, batch32/four bases, direct CSR, uint32 pairs and rebuilt SM86+ CUDA. Unsupported explicit requests raise.
                  - False
                * - current_replay_mode
                  - Omitted/null resolves to ``"record"`` for FP32 temporal carry and inactive (None) for compute carry. Explicit ``"record"`` and approximate ``"recompute"`` both require the FP32 temporal runner; neither changes precision.
                  - None
                * - temporal_pack_spike_checkpoints
                  - Pack binary spike-history checkpoint tensors in the temporal runner.
                  - False
                * - recurrent_weight_scale
                  - 
                  - 1.0
                * - lr_scale
                  - 
                  - 1.0
                * - max_delay
                  - 
                  - 5
                * - pseudo_gauss
                  - 
                  - False
                * - dynamics_mode
                  - Select ``"legacy"`` for recommended training or ``"nest"`` for separately validated compatibility inference/evaluation. NEST training is experimental and is not recommended for use. NEST changes timing, integration, delays, state, and timestamps.
                  - "legacy"
                * - train_recurrent
                  - 
                  - True
                * - train_recurrent_per_type
                  - 
                  - True
                * - noise_seed
                  - 
                  - 0
                * - hard_reset
                  - Reset voltage to ``V_reset`` after a spike when true; use subtractive soft reset when false. Training resolves omitted or null to ``False`` and rejects explicit ``True``. Inference-only and direct-cell defaults are ``False`` in legacy and ``True`` in NEST; existing models retain their built reset setting.
                  - False for training; otherwise mode-dependent
                * - tau_basis
                  - 
                  - <None>
                * - synaptic_basis_weights
                  -
                  - <None>
                * - use_fused_cuda
                  - Use the optional fused CUDA synaptic-current operator. ``True`` requires it; ``"auto"`` falls back to TensorFlow when unavailable.
                  - False
                * - use_pair_projection
                  - Select the recurrent CUDA backward kernel. ``"auto"`` uses pair projection for batch 32 with four basis columns; ``True`` requires it; ``False`` forces the general kernel.
                  - "auto"
                * - use_packed_sm120_backward
                  - Select the packed recurrent backward on SM86 or newer. ``True`` requires float16, batch 32, four basis columns, ``uint32`` compact-pair metadata, and qualified hardware; ``False`` retains the prior pair kernel.
                  - "auto"
                * - use_packed_sm120_external_backward
                  - Select the packed weight-only backward for trainable input populations on SM86 or newer. Fixed inputs build no pair metadata or backward; ``False`` retains the prior external kernel.
                  - "auto"
                * - use_fixed4_input_forward
                  - Use the one-owner input forward for populations with exactly four incoming edges per postsynaptic neuron. Nonqualifying populations retain grouped/general forwarding.
                  - False
                * - use_fused_current_accumulation
                  - Accumulate recurrent and fused spike-input currents through one additive CUDA buffer. Requires fused CUDA currents; current-type inputs retain the TensorFlow addition path.
                  - False
                * - use_direct_csr_recurrent_gradient
                  - Accumulate recurrent gradients in CSR order and restore master-variable order at the full or segmented BPTT boundary. Requires individually trainable recurrent edges and fused CUDA; packed kernels are optional.
                  - False
                * - use_small_batch_recurrent_backward
                  - Experimental local batch reduction for batch sizes 1 through 8 with fused CUDA; independent of pair projection and gradient layout.
                  - False
                * - use_active_row_forward
                  - Opt into active-source-row forwarding for batch sizes 1 through 32 with four basis columns and fused CUDA. Does not pad or change the training batch.
                  - False
                * - track_voltage_penalty
                  - Accumulate a compact neuron-mean voltage penalty at each timestep. Enable only with an online ``VoltageRegularization`` loss.
                  - False
                * - voltage_penalty_mode
                  - Compact voltage penalty: ``"range"`` penalizes voltages outside the normalized range [0, 1], while ``"threshold"`` penalizes distance from threshold.
                  - "range"
                * - return_voltage_sequences
                  - Return neuron-resolved voltage sequences. Setting this to ``False`` requires ``track_voltage_penalty=True``.
                  - True


Setting the Network Model
=========================

Before either training or inference can begin, DPointNet must have instantiated network files that describes the cells, 
synapses, and all the required properties. You can build a network from scratch using the :doc:`BMTK Network Builder <builder>`,
or pre-built models like the ones developed at the `Allen Institute <https://brain-map.org/our-research/computational-modelling>`_
or from other labs.


.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
      
        "networks": {
          "nodes": [
            {
              "nodes_file": "$NETWORK_DIR/glifs_nodes.h5",
              "node_types_file": "$NETWORK_DIR/glifs_node_types.csv"
            },
            {
              "nodes_file": "$NETWORK_DIR/virts_nodes.h5",
              "node_types_file": "$NETWORK_DIR/virts_node_types.csv"
            }
            ],
            "edges": [
            {
              "edges_file": "$NETWORK_DIR/glifs_glifs_edges.h5",
              "edge_types_file": "$NETWORK_DIR/glifs_glifs_edge_types.csv"
            },
            {
              "edges_file": "$NETWORK_DIR/virts_glifs_edges.h5",
              "edge_types_file": "$NETWORK_DIR/virts_glifs_edge_types.csv"
            }
          ]
        }

  .. tab-item:: Python

    .. code:: python

        import numpy




Network requirements
--------------------

instantiating the Network
-------------------------

Networking options
^^^^^^^^^^^^^^^^^^

Network Components
^^^^^^^^^^^^^^^^^^

Combining multiple networks together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filtering for subnetworks
^^^^^^^^^^^^^^^^^^^^^^^^^


Input Stimuli
=============

Spiking Stimulus
----------------

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
          "inputs": {
            "<INPUT_NAME1>": {
              "input": "spikes",
              "module": "<INPUT_MOD1>"
              "node_set": "<virtual_pop_1>",
              "<module_params_1>": <value_1>,
              "<module_params_2>": <value_2>,
              ...
            },
            "<INPUT_NAME2>": {
              "input": "spikes",
              "module": "<INPUT_MOD1>"
              "node_set": "<virtual_pop_1>",
              "<module_params_1>": <value_1>,
              "<module_params_2>": <value_2>,
              ...
            }
          }
        }
      

  .. tab-item:: Python

    TBA




Spike-inputs modules and options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Available "run" options
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
        * - random
          - 
        * - bernoulli_spikes 
          -
        * - poisson_spikes
          - 
        * - lgn_tf
          - 
        * - spikes_files
          - 
        * - custom_spikes_functions
          - 


Building your own inputs module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Initial Conditions
==================

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
            "initial_states": {
                "<NAME1>": {                    
                    "module": "<INIT_MOD1>",
                    "run_on": "all",
                    "<module_params_1>": <value_1>,
                    "<module_params_2>": <value_2>,
                    ...
                },
                "<NAME2>": {                    
                    "module": "<INIT_MOD2>",
                    "run_on": "epoch",
                    "<module_params_1>": <value_1>,
                    "<module_params_2>": <value_2>,
                    ...
                }
           }
        }
      

  .. tab-item:: Python

    .. code:: python



Available modules
^^^^^^^^^^^^^^^^^

.. dropdown:: Available "run" options
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
        * - zero_state
          - 
        * - random_state
          - 
        * - from_input
          - 
        * - cached_states
          - 


Creating your own module
^^^^^^^^^^^^^^^^^^^^^^^^






Training Options
================

Training hyper-parameters
-------------------------

The default training path remains full-sequence BPTT with neuron-resolved voltage
outputs. The performance and memory options below are conservative so existing
configurations retain their previous behavior:

.. list-table:: Training and output options
   :header-rows: 1

   * - option
     - default
     - description
   * - ``gradient_checkpointing``
     - ``False``
     - Enable segmented exact BPTT recomputation.
   * - ``gradient_checkpoint_chunk_size``
     - ``25``
     - Number of timesteps per recomputed chunk when checkpointing is enabled.
   * - ``pack_spike_checkpoints``
     - ``False``
     - Pack the binary delayed-spike state into positive 31-bit words at chunk boundaries.
   * - ``regenerate_initial_state_each_epoch``
     - ``True``
     - Generate fresh configured initial state at each epoch boundary. Set to ``False`` to reuse the initial state across epochs.
   * - ``learning_rule``
     - ``"bptt"``
     - Select standard BPTT or a registered local learning rule.

Segmented exact BPTT retains recurrent state only at temporal chunk boundaries
and recomputes each chunk during the backward pass. Internal Poisson timestep
progression is explicit recurrent state, so recomputation uses the same
stateless random draws as the original forward pass.

.. code:: json

    {
      "training": {
        "gradient_checkpointing": true,
        "gradient_checkpoint_chunk_size": 25,
        "pack_spike_checkpoints": true
      }
    }

``gradient_checkpoint_chunk_size`` must be between 1 and the configured sequence
length. Smaller chunks reduce activation memory but increase recomputation
overhead. The default is 25 timesteps; checkpointing remains disabled unless
explicitly requested.

``pack_spike_checkpoints`` applies only to the first recurrent state, which is
the binary delayed-spike history for the built-in GLIF cell. Every nonzero value
is treated as a spike. Voltage, refractory, ASC, PSC, and replay-safe Poisson
state remain unpacked. Packing is opt-in and has no effect unless segmented
checkpointing is enabled. It reduces checkpoint memory but can add bit-packing
work, so benchmark representative training before enabling it by default.

Compact online voltage regularization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full voltage sequences have shape ``[batch, time, neurons]`` and can dominate
output-gradient memory. If no analysis or learning rule needs neuron-resolved
voltages, DPointNet can instead emit one pre-aggregated penalty value per sample
and timestep:

.. code:: json

    {
      "rnn_cell_params": {
        "track_voltage_penalty": true,
        "voltage_penalty_mode": "range",
        "return_voltage_sequences": false
      },
      "training": {
        "parameters": [
          {
            "loss_functions": {
              "voltage": {
                "module": "VoltageRegularization",
                "penalty_mode": "range",
                "online": true
              }
            }
          }
        ]
      }
    }

The cell and loss ``penalty_mode`` values must match. Online mode supports
``"range"`` and ``"threshold"`` penalties and does not support a core mask.
The defaults are ``online=False``, ``track_voltage_penalty=False``, and
``return_voltage_sequences=True``. Keep those defaults when another consumer,
including a local learning rule, requires full voltages.

Callbacks
---------

Default Callbacks class
^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
          "callbacks": {
            "class": "Callbacks",
            "starting_epoch": 0,
            "callbacks_dir": "training_callbacks_intro_l4_overall_distribution",
            "verbose": "on_step",
            "memory_report": "epoch",
            "epoch_store_weights": "latest",
            "epoch_cache_weights": false,
            "sonata_output_dir": "network.trained_weights.best",
            "losses_table_csv": "losses.csv",
            "performance_table_csv": "performance.csv"
          }
        }
      

  .. tab-item:: Python

    .. code:: python




.. dropdown:: Available "Callback" options
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
          - default
        * - callbacks_dir
          - 
          - callbacks_outputs
        * - starting_epoch
          - 
          - 0
        * - verbose
          -
          - full
        * - time_fmt
          -
          - '%d-%m-%Y %H:%M'
        * - epoch_cache_weights
          -
          - False
        * - epoch_store_weights
          -
          - best
        * - sonata_output_dir
          -
          - trained_weights
        * - losses_table_csv
          -
          - losses.csv 
        * - performance_table_csv
          -
          - performance.csv
        * - memory_report
          - ``epoch`` reports GPU memory at epoch end and once before epoch 1;
            ``step`` additionally reports after each step; ``off`` disables
            memory sampling and peak resets. Console output also follows ``verbose``.
          - epoch

Memory reports separate TensorFlow allocator current/peak allocation, the current
process's driver-reported usage, and whole-device used/free/total memory, in GiB.
Device usage includes other processes; driver process usage includes reservations
and CUDA overhead outside TensorFlow allocation. Differences between these values
are not a fragmentation measurement. Driver values are point-in-time samples, not
peaks.

TensorFlow peak statistics are reset at epoch start. The pre-epoch-1 sample captures
earlier allocation history; epoch 1 includes any first-step tracing/compilation.
Later reports label their reset interval. If a reset fails, the previous interval
label is retained. In ``step`` mode, peaks remain cumulative within that interval,
not per-step peaks. External resets of the same TensorFlow device statistics are
not tracked by the callback.

Unavailable telemetry is shown as ``n/a``. Driver attribution requires an
unambiguous single-GPU mapping or UUID-based ``CUDA_VISIBLE_DEVICES`` without
additional TensorFlow visibility filtering or virtual devices. Ambiguous numeric
multi-GPU mappings and explicit MIG mappings are not guessed. Process usage may
be unavailable when the driver PID namespace differs from the Python process.

Performance CSV output retains ``resident_*_gib`` as whole-device metrics for
compatibility and adds ``process_used_gib`` and ``tf_allocator_peak_scope``.
The latter is a text-valued metric. Default epoch reporting no longer emits
per-step memory rows; select ``memory_report="step"`` to retain them. Each report
reuses one sample for CSV and console output. DPointNet logging has its own
non-propagating logger to avoid duplicate output from configured root handlers.


Building your own Callbacks class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters
----------


Training Inputs
---------------


Initial State
-------------


Loss Functions
--------------

.. dropdown:: Built-in Loss Modules
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
        * - SpikeRateDistributionTarget
          - 
        * - TargetFiringRate
          - 
        * - LowRateFloor
          - One-sided squared firing-rate deficit for preventing low-rate collapse.
        * - VoltageRateFloor
          - Online voltage deficit gated by a detached, accepted-update firing-rate EMA.
        * - OrientationSelectivityLoss
          -
        * - VoltageRegularization
          -
        * - SynchronizationLoss
          -
        * - EMDWeightRegularization
          -






Low-Rate Neuron Rescue
^^^^^^^^^^^^^^^^^^^^^

``LowRateFloor`` is an optional regularizer for neurons whose firing rates fall
below a configured floor. It is independent of legacy/NEST dynamics and uses
spikes, not voltages. It does not replace a firing-rate distribution target and
does not penalize neurons at or above the floor.

For spikes shaped ``[batch, time, neurons]`` and ``rnn.dt`` in milliseconds:

.. math::

    r_i = \frac{1000}{B T \Delta t}\sum_{b=1}^{B}\sum_{t=1}^{T}s_{bti},
    \qquad
    L = \frac{c}{|S|}\sum_{i\in S}\left[\max(0, f-r_i)\right]^2.

Here ``floor_hz`` is :math:`f`, ``cost`` is :math:`c`, and :math:`S` is the
selected neuron set. Rates are pooled across batch and time before applying the
penalty; the denominator includes all selected neurons, not only low-rate ones.
An empty selection returns zero. The whole supplied time window is used, without
implicit trimming. Time length must be positive and statically known. The loss
uses the existing compact temporal reduction, with FP32 batch averaging and
penalty arithmetic, avoiding a full FP32 copy of mixed-precision spike outputs.

Add this entry to a training parameter's ``loss_functions`` mapping:

.. code:: json

    {
      "rate_floor": {
        "module": "LowRateFloor",
        "floor_hz": 0.1,
        "cost": 1.0
      }
    }

The constructor defaults are ``floor_hz=0.1`` and ``cost=1.0``; the loss is never
enabled automatically. ``dt`` must be finite and positive, and floor/cost must be
finite and nonnegative. Either zero floor or zero cost disables the penalty for
nonnegative spikes. A coefficient of 10 scales the same deficit tenfold; it is
not a universal recommendation.

By default all neurons are selected without reading population files. Optional
``neuron_ids`` are unique, nonnegative indices in model spike-column order, not
arbitrary SONATA node IDs. Alternatively, use ``core_mask`` or ``core_radius``
and optionally ``cell_types`` with the existing V1 population helpers and
``data_dir``. Do not combine explicit IDs with core/cell-type selection. For the
200-micrometer V1 core excitatory subset, specify ``core_radius=200.0`` and
``cell_types=["L2/3 Exc", "L4 Exc", "L5 Exc", "L6 Exc"]`` with the appropriate
data directory. Core/cell-type selection preserves the helpers' model ordering.

Choose the subset and floor to avoid forcing biologically appropriate silent
neurons to fire. The pooled rate quantum is ``1000 / (batch * time * dt)`` Hz.
The loss gives an upward rate gradient below the floor, but rescue of network
weights still requires a nonzero surrogate-gradient path through the cell; this
is not guaranteed for deeply subthreshold neurons. Assess rate distributions,
silent fractions, and the original fitting objective, not just this penalty.

Configured losses participate in ordinary training and validation totals. A
selection score that excludes rescue must be implemented explicitly by the
experiment; this module does not silently alter checkpoint-selection policy.
No project-specific registration call is required.


Online Gated Voltage Rescue
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``VoltageRateFloor`` is a separate, **default-disabled** BPTT loss for direct
subthreshold voltage credit. It does not replace ``LowRateFloor`` or the existing
range/threshold ``VoltageRegularization``. It selects **all model neurons**,
including inhibitory neurons, without population files or implicit trimming.
Subset arguments currently raise an explicit error.

.. math::

    g_i = \operatorname{stopgrad}\left[
      \operatorname{clip}\left(1-\frac{\bar r_i}{f},0,1\right)\right],
    \qquad
    L = \frac{c}{BTN}\sum_{b,t,i}
      g_i\,a_{bti}\,[\max(q-v^{\mathrm{pre}}_{bti},0)]^2.

Voltages are in the cell's normalized units (threshold one), ``cost`` is
:math:`c`, ``target`` is :math:`q`, ``floor_hz`` is :math:`f`, and
:math:`a` excludes refractory timesteps. The denominator is all neurons, batch
members and timesteps, **not** the number of active or gated entries. NEST uses
the voltage before this timestep's reset and the incoming refractory counter;
legacy uses its pre-threshold voltage and updated refractory counter. Neither
definition changes spikes, resets, dynamics, surrogate derivatives or recurrent
spike adjoints. Fused state/current and direct-CSR/packed backward dispatch remain
available; no dense voltage sequence or full-FP32 spike-gradient expansion is
introduced.

Rebuild the CUDA operators before enabling this loss with fused NEST state.
Its forward operator adds the default-false ``emit_pre_reset_voltage`` attribute:
when enabled, the existing threshold channel carries the exact pre-reset value,
and the wrapper subtracts threshold for spike/backward dispatch. No extra dense
channel or new packed kernel is allocated. This avoids cancellation from
reconstructing small voltages by adding threshold back after subtraction.
The default operator/wrapper output semantics remain unchanged; opting in with
an older binary raises a rebuild error rather than silently disabling fusion.

Add the same entry to each participating training parameter's ``loss_functions``:

.. code:: json

    {
      "voltage_floor": {
        "module": "VoltageRateFloor",
        "cost": 1.0,
        "target": 0.9,
        "floor_hz": 0.1,
        "ema_decay": 0.95
      }
    }

These are opt-in defaults, not a qualified universal rescue recipe. Cost must be
nonnegative, floor and ``rnn.dt`` positive, and decay in ``[0, 1)``; all values
must be finite. ``enabled: false`` creates no online channels or history state.
Programmatic callers must instantiate this loss before building the RNN (and
before constructing other losses that build it). The normal JSON loader registers
its channels first, irrespective of loss ordering. Identical effective
configurations, whether JSON or programmatic, share history and one accumulator
channel. Different costs or rate-gate settings retain independent channels.

The FP32 accumulator adds only one scalar per trial and loss configuration to
the recurrent state. Existing full-voltage outputs and compact range penalties
are preserved. Set ``return_voltage_sequences: false`` and
``track_voltage_penalty: true`` to retain compact spike/range outputs alongside
the new accumulator. Exact checkpoint recomputation carries the accumulator
across chunks; a new logical extractor rollout clears only this accumulator.
History reads and all loss evaluation are replay-pure.
Enabled cell and recurrent-wrapper boundaries preserve the accumulator in FP32,
including under Keras 3 mixed precision. Physical floating states retain their selected policy: with
``state_precision="selective"``, voltage and ASC remain FP32 while synaptic and
history state remain FP16. The accumulator is never narrowed to FP16.
The native training engine supports this loss with
``temporal_gradient_precision="float32"`` and its default ``current_replay_mode="record"``.
The explicit ``"recompute"`` mode reprojects currents and remains approximate
when atomic current reductions are nondeterministic. Both modes carry the
online accumulator through the FP32 temporal adjoint; optional
``use_fused_recurrent_accumulation`` remains independent and default-false.
Only per-chunk spike-output cotangents are widened, not the full compact
sequence. Fused NEST capture preserves the selected Gaussian/triangular
surrogate and reset/ASC derivative settings.
Existing cached physical initial states and randomized-state specifications may
omit the new accumulator: it is initialized to zero. Such initial states never
initialize or overwrite the separate firing-rate history.

**Startup is measured, not an all-silent prior.** Uninitialized gates are zero.
The first accepted optimizer update commits firing rates measured on that
forward pass and initializes the gates; it receives no voltage rescue itself.
Alternatively call ``loss.initialize_rates(rates_hz)`` with a finite,
nonnegative, model-ordered all-neuron vector, before or after model build, to
seed measured baseline history explicitly. Subsequent accepted updates use
``ema_decay * previous + (1 - ema_decay) * measured``. Gates are computed only
when history is committed, detached, and held fixed throughout forward/backward
and replay. Validation, replay and rejected dynamic-loss-scale steps do not
change history. Zero-cost enabled losses still collect history.

Parallel training pools spike counts across the entire concatenated model batch
(and replicas) with batch/time weighting, not an equal mean of condition rates.
The voltage penalty contributions are also batch weighted. Per-condition loss
entries include a factor equal to the number of conditions to compensate for
the native reported-total condition average; the reported rescue contribution
is therefore the mean over the combined batch, not that mean divided again by
the number of conditions. Accumulated training and validation use identical
weights. Series updates commit
once after each condition; ``series_accumulate`` pools counts and commits once
after its combined update. Rate history is independent of OSI/DSI normalizers.
Native SGD, Adam and ExponentiatedAdam, including Keras 2/3 dynamic loss scaling,
use accepted-update accounting. Non-BPTT learning rules are explicitly rejected.
History, initialized flag, gate, accepted-update count and numeric configuration
are nontrainable cell weights, included in model weight saves and
``tf.train.Checkpoint(model=rnn.model, optimizer=...)``. Reconstruct the same loss
channel configuration before restoring a checkpoint.

This is a surrogate-gradient research tool, not a strict-local rule or evidence
of firing-rate recovery. Monitor original fitting losses, rates and silent
fractions, consider biologically appropriate silence, and record the full
configuration and baseline-rate provenance. Large experimental coefficients
(for example 100) are not defaults. NEST training remains experimental and is
not generally recommended. Configured rescue participates in validation totals;
an alternative checkpoint-selection objective must be an explicit experiment
policy.


Training Output
---------------




Running Inference
=================


Running an Inference
--------------------


Results/Output
--------------

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
            "output": {
                "output_dir": "$OUTPUT_DIR",
                "log_file": "$OUTPUT_DIR/log.txt",
                "log_level": "INFO",
                "spikes_file": "$OUTPUT_DIR/spikes.h5",
                "overwrite_results": true
            }
        }
      

  .. tab-item:: Python

    .. code:: python
