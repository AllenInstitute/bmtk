# DPointNet parity branch: settings and agent handoff

## Scope and provenance

Branch: `feature/dpointnet-javier-parity`, based on consolidated training commit
`58a57645840075a4f4da589787492924beb502bb`. This integrates the tested v11 source
snapshot (stages S1-S16, 2026-09-30), not just the final S16 patch.
No shared editable installation or scientific environment is changed by this commit.

The subsequent2026-09-30 follow-up adds [variable-batch acceleration](dpointnet_variable_batch.md)
and [startup preprocessing](dpointnet_startup.md). The original batch32 overlay
below remains valid. For batches below32, follow the variable-batch guide rather
than forcing the batch32-only packed flags. Device Poisson remains default-off
because changing its algorithm changes seeded realizations.

Javier-derived kernels, grouped EMD and optimizer work retain source attribution to
`v1_model_utils` at commit `2c52ec10` in their implementation comments. The user
confirmed contractual permission to adapt this code on 2026-09-29.

The measurements qualify a throughput implementation, not NEST training convergence
or a universal learning-rate optimum. Keep legacy dynamics for new unrelated
projects; do not silently switch an existing experiment's dynamics or precision.

## Precision profiles

- **Default (FP16 temporal backward):** selective forward state with ordinary
  per-state temporal gradients (`temporal_gradient_precision="compute"`, which is
  also the constructor default) and native dynamic loss scaling. Use the explicit
  overlay below for the qualified NEST/SM86/batch32 topology.
- **Opt-in accuracy reference:** selective forward state with
  `temporal_gradient_precision="float32"` and `current_replay_mode="record"`.
  Use it when a task depends on very small long-lag credit, when reverse accuracy
  is itself under study, or to check an FP16 result. Keep both public
  `use_packed_sm120_*` flags **false**; the internally selected FP32 recurrent
  kernel is distinct from the FP16 packed kernels. It costs about 1.5–1.8x update time.
- `current_replay_mode="recompute"` is an explicitly approximate alternative:
  atomic current projection need not reproduce the original forward exactly.

The existing [training guide](autodocs/source/dpointnet_guide.rst) describes general
APIs. Most parity accelerators are opt-in constructor flags. ExpAdam now defaults
to `jit_compile=true` on its eligible path; set it explicitly and validate XLA
availability on a new environment.

## Qualified default overlay

[dpointnet_parity_overlay.json](dpointnet_parity_overlay.json) is a **partial config**:
merge these run/cell/training keys into a complete project-owned config. It contains
no data paths, input definitions, targets or permission to launch training.

Qualifications:

- RTX 3090 / SM86; TensorFlow 2.21, Keras 3; effective model batch32; four synaptic
  bases; compact uint32 CSR/pair metadata; per-edge trainable recurrent weights.
- The benchmark used parallel conditions **24 evoked +8 spontaneous**, one
  optimizer update, 66,658 neurons, 1-ms timestep and 500-ms sequences.
  Set `training.parameters` batch sizes accordingly. A single-condition config
  can use batch32, but is not the measured scientific workload.
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

## Build and verify in a separate checkout

Pin the actual commit, then record the source override in the project's execution
contract. Never relink an immutable scientific environment. With the engineering
prefix already available:

```bash
cd /path/to/this/checkout
export PYTHONPATH="$PWD"
export PATH=/local2/mmroot/envs/rule_search-dpnet/bin:$PATH
unset TF_GPU_ALLOCATOR
python -c 'import bmtk; print(bmtk.__file__)'
DPOINTNET_CUDA_ARCHS=86 python -m bmtk.simulator.dpointnet.custom_ops.build
python -m pytest -q tests/simulator/dpointnet
```

Rebuild both operators: NEST type-index inputs changed from int32 to int64.
Do not reuse older binaries. Keep FP32 masters, canonical neuron/edge order,
constraints and recurrent/named-input shadow refresh. Do not prune silent-neuron
spike adjoints.

## Measured results and limits

RTX3090, default BFC, 20-GiB logical cap, three excluded warmups plus 20
synchronized native updates, paired ABBA:

| Run | Median seconds/update |
|---|---:|
| Javier A | 2.6463 |
| DPointNet A | 2.6467 |
| DPointNet B | 2.6675 |
| Javier B | 2.6494 |

Mean of medians: **2.657 vs 2.648 s (1.003x)**. Separate same-session precision
comparison: ordinary2.665s/10.56GiB, FP32-record4.761s/12.06GiB,
FP32-recompute4.057s/12.04GiB. Memory is the TensorFlow peak over a separate
restored update, not driver allocation or total system memory.

The last gains removed per-step host/device transfers (type-index table, CPU
Poisson operands, output-list indices) and parallelized voltage-penalty reduction.
This does **not** mean int32 is always host-resident or forbidden in CUDA:
placement depends on the operation and graph. Diagnose actual transfers first.

Full-network late-window gradient/update comparisons at100/250/500ms with native
loss scale32768 found FP16-vs-FP32 differences comparable to same-precision repeats.
Those were **separate forward executions**, not an identical-forward oracle:
atomic-order variability obscures smaller precision errors. Initial-state gradient
norms alone cannot rule out spurious small tails. The prior controlled small-network
FP16 rounding plateau is not disproved, and no learning/convergence test establishes
that FP16 is safe for every delayed-credit task. FP16 is therefore the default,
with FP32-record as the check for tiny long-lag signals and precision qualification.

The tested snapshot passed1567 GPU tests (6 skipped),958 CUDA-disabled tests
(615 skipped), and922 actual Python3.8/Keras2 CPU tests (651 skipped).
Logs, raw timing/F1/F2 artifacts and stage reports are retained locally under
`/local2/results/dpointnet_rule_search/javier_parity_20260929/stages/S16`.
These are local test results, not GitHub CI or L40S/A100 results.

Integration verification: both SM86 operators rebuilt in the branch checkout;
317 targeted acceleration, training, EMD and optimizer tests passed (2 skipped).
Runtime source and tests match v11, apart from whitespace cleanup in the CUDA file.
