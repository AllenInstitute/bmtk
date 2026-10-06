# DPointNet performance configuration

## Scope

This guide describes opt-in acceleration and precision settings for advanced
training users. Library defaults favor compatibility; an accelerated profile must
match the selected hardware, topology, precision and training objective.

For batches below 32, follow the [variable-batch guide](dpointnet_variable_batch.md)
rather than forcing the batch-32-only packed flags. Device Poisson remains
default-off because changing its algorithm changes seeded realizations.

Javier-derived kernels, grouped EMD and optimizer work retain source attribution to
Javier Galvan's `JavierGalvan9/V1_GLIF_model`, `v1_model_utils` at commit
`2c52ec10c1eee409ddf900f8a5b8460cf9c46d24`, in their implementation comments.

**Use legacy dynamics for training. NEST training is experimental and is not
recommended.** Throughput and gradient checks do not establish training
convergence. NEST is an opt-in compatibility mode requiring separate validation
of the intended network and outputs. Do not silently switch an existing
experiment's dynamics, precision or random sampler.

## Precision profiles

### Automatic accelerator selection

Set `"acceleration_profile": "auto"` in `rnn_cell_params` to select compatible
execution accelerators at model build time. Omitting it preserves existing
defaults. Explicit individual flags take precedence and retain their normal
validation; an unsupported explicit `true` is not silently downgraded.

```json
{
  "rnn_cell_params": {
    "acceleration_profile": "auto",
    "dynamics_mode": "legacy"
  }
}
```

Selection uses GPU compute capability (SM), the loaded operators' architecture
manifests and entry points, compute/master/temporal dtypes, batch size and basis
width. CUDA toolkit version alone does not identify GPU capabilities.
TensorFlow/CUDA/cuDNN build versions and selection reasons are logged and
available as `rnn.acceleration_report`. This records selected configuration,
not proof that every topology-specific kernel executed.

Compatible generic CUDA routes remain eligible on SM70/75/80; FP16 packed
backward and recurrent accumulators require SM86+. Multiple visible GPUs or
missing/incompatible operators select the documented TensorFlow fallback with
a warning. General NEST/Pascal is not automatically admitted. Fixed-four inputs
and packed metadata retain per-connectivity validation/fallback. Recurrent
accumulation additionally requires an explicitly selected direct-state loop or
FP32 replay route, direct CSR and trainable per-edge weights.

The profile does not change dynamics, reset semantics, precision, temporal
credit, loss functions, optimizer, allocator, random sampler or scientific
parameters. It neither rebuilds/installs libraries nor establishes workload
memory feasibility, convergence or cross-GPU equivalence. Rebuild and qualify
the intended environment normally. Standalone cells and external runners can
use `bmtk.simulator.dpointnet.acceleration.resolve_acceleration_options` with
their actual dtypes, batch and basis width; runner-only accelerators remain the
runner's responsibility. The report includes both requested and resolved flags.

External weight-carrier runners can additionally use
`resolve_weight_carry_options` and `project_weight_carry` from the same module.
The native accumulator is gated independently on SM86+: disabling Javier alone
does not make its kernel compatible with SM70/75/80. The compatible generic
route combines direct-CSR values/gradients with a TensorFlow identity carrier,
retaining live recurrent spike credit, including silent-neuron adjoints.
Explicit stopped external inputs retain a zero-scaled unused spike VJP on the
generic route because its direct-CSR contract requires that VJP. Optional native
weight-only projection additionally requires stopped-input eligibility and the
compatible rebuilt operator. Explicit incompatible carrier requests fail.
External runners must also apply the resolved external-packed/fixed-four flags
to every input projection and wire project-specific smoothing compatibility;
the cell resolver cannot control those independent runner surfaces.

- **Throughput profile (FP16 temporal backward):** selective forward state with ordinary
  per-state temporal gradients (`temporal_gradient_precision="compute"`, which is
  also the constructor default) and native dynamic loss scaling. Apply only
  accelerators whose hardware and topology qualifications are satisfied.
- **Opt-in accuracy reference:** selective forward state with
  `temporal_gradient_precision="float32"` and `current_replay_mode="record"`.
  Use it when a task depends on very small long-lag credit, when reverse accuracy
  is itself under study, or to check an FP16 result. Keep both public
  `use_packed_sm120_*` flags **false**; the internally selected FP32 recurrent
  kernel is distinct from the FP16 packed kernels. This accuracy reference can
  require more update time and memory; measure its cost on your workload.
- `current_replay_mode="recompute"` is an explicitly approximate alternative:
  atomic current projection need not reproduce the original forward exactly.

The existing [training guide](autodocs/source/dpointnet_guide.rst) describes general
APIs. Most accelerators are opt-in constructor flags. ExpAdam defaults
to `jit_compile=true` on its eligible path; set it explicitly and validate XLA
availability on a new environment.

## Batch-32 benchmark overlay

[dpointnet_parity_overlay.json](dpointnet_parity_overlay.json) is a **partial config**:
it contains no data paths, input definitions or targets. It records a specific
NEST throughput benchmark, **not a recommended NEST training recipe**. Do not
merge it unchanged into a new training experiment. Select legacy dynamics and
your own scientific settings, then apply compatible acceleration keys to a
complete configuration.

Qualifications:

- RTX 3090 / SM86; TensorFlow 2.21, Keras 3; effective model batch32; four synaptic
  bases; compact uint32 CSR/pair metadata; per-edge trainable recurrent weights.
  These describe the benchmark qualification, not universal hardware requirements.
- The benchmark used parallel conditions **24 evoked +8 spontaneous**, one
  optimizer update, 66,658 neurons, 1-ms timestep and 500-ms sequences.
  The overlay's `training.parameters` batch sizes reflect that split. A
  single-condition config can use batch 32, but is a different workload.
- Static type-indexed NEST dispatch requires coefficients to be identical within
  each declared cell type; its validation must succeed. Do not force it when
  coefficients vary within type.
- Fixed-four input forward requires the corresponding four-edge topology;
  uniform-delay projection requires uniform delays in that input group.
- Packed SM86 performance is **not A100/SM80 qualification**. Do not force the
  public FP16 packed flags on unsupported GPUs; record the supported fallback.

For every `EMDWeightRegularization` loss, set:

```json
{
  "use_grouped_custom_gradient": true,
  "use_javier_grouped_emd": true,
  "deduplicate_within_graph": true
}
```

Deduplication is limited to one training loss/gradient evaluation. Parallel
conditions can share the calculation; series conditions use independent scopes
so each update reads current weights and retains its EMD gradient. Direct loss
calls outside the trainer recompute rather than caching across unrelated tapes.
The cache is owned by the evaluation scope and cleared on exit, including
exceptions; regularizers do not retain old scopes or graphs across retracing.
Explicit-state rollout coefficients and noise-seed snapshots are also temporary:
they are restored after preparation, execution, or tracing failures. Rejected
FP32 masked/time-major calls are validated before installing those snapshots.

Retain all scientific losses. The measured V1 loss set included firing-rate
distribution targets, online range voltage regularization, synchronization,
EMD weight regularization and evoked orientation selectivity. No rescue term was
added. LR0.003, surrogate gain0.05/width0.28, detached reset and attached ASC are
the matched benchmark settings, **not proven optimal learning defaults**.

The overlay disables optional signature/mean-rate metrics for timing parity;
enable desired monitoring for real training and include its cost. It does not
disable losses. Inputs were prepared outside the native-update timer. BKG remains
stateless Poisson in the time loop. Timed updates restored the same inputs,
weights, optimizer and RNG state; this is not a trained-network timing result.

## Build and verify

Activate an environment containing BMTK's dependencies, TensorFlow, a CUDA
toolkit and the test dependencies. From the source checkout you intend to use:

```bash
cd /path/to/this/checkout
export PYTHONPATH="$PWD"
unset TF_GPU_ALLOCATOR
python -c 'import bmtk; print(bmtk.__file__)'
DPOINTNET_CUDA_ARCHS=86 python -m bmtk.simulator.dpointnet.custom_ops.build
python -m pytest -q tests/simulator/dpointnet
```

The architecture override above targets SM86; choose the target for your GPU
instead. For a multi-architecture HPC artifact, use
`DPOINTNET_CUDA_ARCHS="61 70 75 80 86 89 90"` with a compiler supporting every
requested target. Do not silently omit unsupported targets; declare separate
toolchain/artifact profiles when necessary.
Rebuild both operators after source, TensorFlow or CUDA changes:
NEST type-index inputs use int64, so older int32 binaries are incompatible.
Record the source revision and resolved import path. Keep FP32 masters, canonical neuron/edge order,
constraints and recurrent/named-input shadow refresh. Do not prune silent-neuron
spike adjoints.

### Optional auditory weight-only accumulation

`custom_ops.csr_spike_ops.fused_recurrent_weight_carry` accepts
`compute_spike_gradient=False` for fixed external spikes whose adjoints are
already stopped. It preserves weight accumulation while avoiding the FP16
Javier input-spike projection/VJP workspace. This requires rebuilt compatible
operators, FP16 operands and `use_javier_batch32_backward=True`.

The default remains `True`: recurrent spike credit, including silent-neuron
credit, is retained. Existing default wrapper calls also remain compatible
with the old accumulator signature. Requesting the new option against an old
binary raises a rebuild error. Automatic acceleration never disables spike
credit or assumes an input is stopped; the input-owning runner must make that
explicit scientific/graph decision.

## Precision and measurement limits

FP16 can lose very small long-lag signals. Full-network comparisons using separate
forward executions cannot rule this out: atomic-order variability can obscure
smaller precision errors. Initial-state gradient norms alone cannot rule out
spurious small tails, and passing numerical tests does not establish learning
quality for every delayed-credit task. Use FP32 record replay to check workloads
that depend on weak or long-horizon credit.

For performance comparisons, match hardware, inputs, objective, optimizer,
precision and output mode. Exclude tracing/compilation and warmup, synchronize
timed outputs, and distinguish native-update timing from end-to-end timing with
fresh inputs. Report TensorFlow peak allocation separately from driver reservation.
Do not extrapolate untrained-network timings to trained higher-activity networks.

Diagnose actual host/device transfers before changing integer dtypes or placement:
int32 is not universally host-resident or forbidden in CUDA. Keep source and
environment revisions with your measurements rather than treating one benchmark
as a general speed guarantee.
