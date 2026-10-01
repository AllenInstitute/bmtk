# Pascal GPU compatibility

This follow-up to `7bf7855` adds Pascal-compatible CUDA atomics/warp grouping
and LGN spatial filtering. It does not change neuronal dynamics, losses,
configured precision, background distribution or installed dependencies.

## Build and settings

Build the selected source in a project-owned environment with
`DPOINTNET_CUDA_ARCHS=61 python -m bmtk.simulator.dpointnet.custom_ops.build`.
For a multi-target binary, use a space-separated list such as `"61 86"`, not
commas. SM61 is not added to the default architecture list.

Pascal does not satisfy the SM86+ accelerator contracts. Set these flags false:

```json
{
  "use_fused_recurrent_accumulation": false,
  "use_javier_recurrent_vjp": false,
  "use_packed_sm120_backward": false,
  "use_packed_sm120_external_backward": false
}
```

Keep other flags only when their independent hardware/topology qualifications
are met. No precision or dynamics switch is required by this patch. Preserve
the selected scientific protocol; legacy remains the unrelated-project default.

## Compatibility implementation

Scalar half accumulation retains native atomicAdd on SM70+ and uses TensorFlow's
CAS-based helper below SM70. Paired half2 accumulation retains its native SM60+
path. Warp grouping retains match-any on SM70+; older GPUs use shuffle/ballot
matching. Duplicate target groups and silent-neuron adjoints are preserved.

LGN pre-Volta architecture selection avoids cuDNN spatial convolution using
bounded extract-patches and matrix multiplication on GPU. SAME padding, filter
weights, normalization and canonical LGN row order are unchanged. The temporary
patch budget is approximately four million elements; large movies are processed
in sequential row/batch chunks rather than materialized as one patch tensor.
The original multi-channel temporal filter is retained to preserve its reduction
order. Single-channel temporal filtering uses the explicit helper because
TensorFlow otherwise reroutes it through unsupported cuDNN Conv2D.

On pre-Volta, firing-rate evaluation uses a non-XLA TensorFlow graph because XLA
can lower the filters back into cuDNN. Modern GPUs keep their original
convolution/XLA paths. This is hardware compatibility, not a blanket speed claim
or a fallback that moves LGN computation to CPU.

## Measured scope

Actual GTX1080Ti (SM61), TensorFlow2.21/CUDA12.9/cuDNN9.25,66,658-neuron V1,
500ms, effective batch16(12evoked+4spontaneous), selective forward/ordinary FP16
temporal gradients, chunk25, original Poisson BKG, default BFC and fresh prefetched
LGN inputs: three finite applied optimizer updates passed. Final-source post-trace
times were8.74/10.24s including fresh input work; first update was51.09s including
tracing. An earlier smoke measured8.88/10.29s and54.28s, respectively.
These are two smoke timing samples, not a qualified20-sample performance benchmark.

Final-source TF allocation peaks were7.83/7.86/7.80GiB per update, including input
generation and prefetch; the earlier smoke reached8.01GiB. Driver process
reservation was about10.43GiB on an11GiB card,
leaving limited headroom. Short smoke feasibility does not establish safe memory
for trained higher-activity networks, longer sequences, other losses or long runs.
TitanXp was not separately tested; do not present it as measured capacity.

Real LGN normalization kernels7x7through27x27,500-frame packed spatial filtering
and574/314-tap temporal filters passed the existing1e-6 CPU-reference gate with
GPU placement required. A rejected all-explicit temporal candidate failed that
gate; tolerances were not loosened. Small gradient/chunk/warp-boundary tests
cover independent reference values and gradients. Nonlinear chunk-boundary
gradients use an FP64 synthetic oracle to isolate indexing from FP32 reduction
cancellation; separate FP32 value/input-gradient tests remain. Production
precision and1e-6 tolerances were not changed.

Complete final-production-source regression suites passed1735GPU tests
(26skipped, on RTX3090/SM86),966CPU/Keras3 tests and966actualPython3.8/TF2.13/Keras2 tests
(795skipped each). The subsequent test-only oracle precision correction passed
all18LGN tests on Pascal and both CPU/Keras versions. No production-code changes
followed the complete suites. No physical multi-GPU or TitanXp result is claimed.

The2026-10-01 combined-branch audit onGTX1080Ti passed68focused alpha-basis,
LGN and recovery tests, but its full GPU suite did not pass:1502passed,
160failed,140skipped. Early failures included the fused NEST forward kernel
requesting too many launch resources; later failures included additional GPU
errors. This does not qualify general NEST execution on Pascal. The earlier
legacy batch16/LGN smoke and SM86 suite are narrower, separate qualifications;
do not interpret them as a complete Pascal regression pass.

Full validation and authoritative memory/timing receipts are recorded in
`/local2/results/dpointnet_rule_search/pascal_bs16_20260930/FINAL_REPORT.md`.
Pin the published revision and rebuild operators before adoption; existing
environments and experiment source checkouts are not upgraded automatically.