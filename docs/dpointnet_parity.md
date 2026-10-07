# DPointNet performance configuration

## Scope

This guide describes the recommended automatic acceleration workflow for new
projects and explicit precision settings. Library defaults favor compatibility;
an accelerated profile must
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

## Recommended workflow for new projects

Build both operators for the full supported GPU pool, then use
`"acceleration_profile": "auto"` rather than maintaining GPU-specific lists of
accelerator flags. This is the recommended new-project starting point; omitting
the profile still preserves library defaults and existing experiments.

1. Pin the consolidated fork's qualified source, for example
   `94f90d70e6353af54bf6205708a7e2b75e85a534` on
   `shixnya/bmtk:feature/dpointnet-training-inputs-consolidated`, in the project's
   dependency contract. Verify the actual BMTK import path.
2. In the intended TensorFlow/CUDA environment, check that `nvcc --list-gpu-code`
   supports every requested target and build:

   ```bash
   DPOINTNET_CUDA_ARCHS="61 70 75 80 86 89 90" \
     python -m bmtk.simulator.dpointnet.custom_ops.build
   ```

   This covers GTX1080Ti/TitanXp, V100, RTX8000, A100, RTX3090, L40S and H200,
   with PTX for SM90. Compile on an allocated compute node for HPC. If the
   compiler cannot cover the list, report the mismatch; do not silently omit
   older GPUs. Binaries remain TensorFlow/CUDA ABI-specific. Compilation does
   not qualify every GPU or authorize its use.
3. Add the automatic profile to your otherwise unchanged configuration:

   ```json
   {
     "rnn_cell_params": {
       "acceleration_profile": "auto"
     }
   }
   ```

   For the eligible direct-loop BPTT route, explicitly select
   `"use_direct_state_rnn_loop": true` as well. Automatic acceleration does not
   choose this route, precision or scientific parameter sharing for you.
   Preserve your per-type/per-edge recipe; recurrent native accumulation
   requires trainable per-edge weights, direct CSR and four bases.
4. Inspect `rnn.acceleration_report`, including fallback/disabled reasons,
   and smoke-test real optimizer updates, restoration and memory on the
   intended GPU. External runners must wire the shared resolver and their own
   input/carrier surfaces; setting a cell flag alone cannot control them.

**Initial build cost:** approximately13-14minutes for both full-target operators
on the tested8-CPU HPC allocation, estimated from job-phase timings. This excludes
environment setup, queue waiting and model initialization/first-step tracing.
Reuse the binaries across supported GPUs and batches in a compatible environment;
rebuild after CUDA-source, TensorFlow ABI or CUDA-toolchain changes.

Let the resolver handle architecture, loaded operators, dtype, topology and
batch restrictions. Explicit individual flags override it, so do not carry
old manual accelerator lists into a new automatic profile unless they are
intentional, documented exceptions. The goal is the fastest qualified route
for the selected workload, not a guarantee of maximum speed on arbitrary GPUs.

Current automatic native policy admits RTX8000/SM75 ordinary FP16 temporal
backward and existing SM86+ paths. Pascal remains an explicit opt-in;
V100/SM70 and A100/SM80 remain conservative/generic pending automatic-policy
qualification. RTX8000 FP32 compute/replay remains generic. Batch16 can use
the native accumulator without batch32-only packed backward; see the
[matched RTX8000 batch comparison](dpointnet_variable_batch.md#matched-rtx8000-batch1632-execution-check).
Use batch32 where the selected workload fits and throughput is the priority;
batch16 is a measured lower-memory alternative, not an automatic batch change.

Do not migrate running/pinned experiments, change scientific settings or
force an unqualified path merely because the fat binary contains its SM target.

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

Compatible generic CUDA routes remain eligible on SM70/75/80. Automatic native
selection admits SM75 (RTX8000) ordinary FP16 temporal backward and the existing
SM86+ paths. SM70/V100 and SM80/A100 retain generic automatic selection while
their portable qualifications are pending. Pascal remains an explicit opt-in.
Multiple visible GPUs or
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
Rebuilt native operators support SM61+ independently of the narrower automatic
policy. External carrier selection also checks the admitted cell route:
compatible hardware alone does not turn on native accumulation. The compatible generic
route combines direct-CSR values/gradients with a TensorFlow identity carrier,
retaining live recurrent spike credit, including silent-neuron adjoints.
Explicit stopped external inputs retain a zero-scaled unused spike VJP on the
generic route because its direct-CSR contract requires that VJP. Optional native
weight-only projection additionally requires stopped-input eligibility and the
compatible rebuilt operator. Explicit incompatible carrier requests fail.
External runners must also apply the resolved external-packed/fixed-four flags
to every input projection and wire project-specific smoothing compatibility;
the cell resolver cannot control those independent runner surfaces.

### Portable preview adoption

Use the fork's `feature/dpointnet-training-inputs-consolidated` branch and pin
the exact commit in each new project's dependency contract. This is an
engineering preview, not an upstream release or a migration of existing
experiments. Keep existing project/environment/checkpoint pins unchanged.

Follow the [recommended multi-architecture build workflow](#recommended-workflow-for-new-projects)
in the intended TensorFlow/CUDA environment, including on an allocated compute
node for HPC.

Both operators must be rebuilt when adopting the portable CUDA changes;
previous fat binaries can contain all SM targets but still have the old SM86
eligibility restriction. Binaries are environment/ABI-specific, not universal
across TensorFlow or CUDA versions.

For qualified RTX8000 FP16 per-edge BPTT, select your scientific recipe and add:

```json
{
  "rnn_cell_params": {
    "acceleration_profile": "auto",
    "use_direct_state_rnn_loop": true,
    "train_recurrent_per_type": false
  }
}
```

This explicitly selects the execution route, not a different dynamics model,
precision, sampler, loss or optimizer. Native accumulation supports batch1..32;
packed gradients still require batch32, four bases and compatible compact
metadata. Batch8/16 keeps the accumulator but disables packed-only paths.
RTX8000 FP32 temporal replay remains generic in automatic selection.
The example presupposes per-edge training: retain per-type training if your
scientific recipe uses it, rather than changing parameter sharing for speed.
Automatic selection then leaves per-edge accumulation disabled.

The initial bounded 66,658-neuron V1+LGN comparison completed fresh and
epoch64-restored baseline/native runs, each with3warmups and20measured updates.
RTX8000 BS32(24+8) measured28.51/28.57s baseline versus5.01/5.87s native,
with native TF peaks13.01-13.04GiB. TitanXp BS16(12+4), using explicit flags,
measured9.68/10.05s versus8.46/8.75s, with native peaks8.65-8.70GiB.
These are acceleration-bundle timings, not isolated-kernel or same-batch
cross-GPU comparisons. They do not establish convergence, long-run memory
safety, arbitrary-topology support or bitwise-equivalent complete updates.
The separate exact-auto RTX8000 release gate passed at executable revision
`9dcd65614bf2fe5b144c53e2abc7fa35c2b12ed6`: full GPU suite1925passed/28skipped,
focused77passed/2skipped, and all nine weight-only regressions executed in both.
Complete CUDA-disabled Keras3 and actual Python3.8/TF2.13/Keras2 suites each
passed1149tests/804skips. Both operators were rebuilt for the full target list.
Fresh and strictly epoch64-restored exact-auto V1+LGN BS32 cases each applied
23updates, with actual accumulator/packed graph evidence and matching starting
masters/LGN inputs against the earlier explicit route. Their20sample medians
were5.03/5.85s and TF peaks12.97/13.57GiB. Peak allocation varies between these
whole-model runs; unchanged accumulator buffers do not guarantee identical
TensorFlow allocator peaks. No rejected updates or initialization fallbacks
occurred; strict checkpoint roundtrips passed. This remains bounded execution
qualification, not a convergence or universal workload-memory guarantee.

GTX1080Ti BS16 failed GPU-memory allocation in the baseline route before native
comparison. The separate BS8(6+2) fresh/trained comparison passed all four
scenarios:6.95/7.16s baseline versus5.45/5.66s native, with native TF peaks
about6.21/6.35GiB. This is a different batch protocol, not recovery of the
failed BS16 qualification. V100/A100/L40S
portable matrix results remain pending. Do not remove compatible/reference
backends or silently enable experimental settings on those projects.

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
