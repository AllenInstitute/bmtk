# Variable-batch acceleration and device Poisson sampling

This guide describes acceleration for advanced users choosing effective model
batch sizes and an optional device-resident background sampler. Select compatible
options for your hardware and topology. Do not change an existing experiment's
source or random sampler in place.

## Recurrent execution

The pair-projected backward and FP32 recurrent accumulator support every batch
from 1 through 32, not only batches 8 and 16. Power-of-two tiles mask samples
outside the actual batch. Batch sizes above 32 retain the pre-existing general
fallback; fused recurrent accumulation still rejects them.

For batches below 32, the existing `use_javier_recurrent_vjp` option selects
scaled-half pair projection, tiled spike gradients and event-sparse weight
gradients. The scale is a power of two; projection rounding remains FP16 while
master weights and accumulated weight gradients remain FP32. Silent neurons
retain their spike adjoints. The unscaled path retains FP32 projections.

The carrier wrapper accepts forward acceleration options. The cell uses
them for variable batches; batch 32 preserves its established carrier-forward
dispatch. Retain `use_fused_recurrent_accumulation`,
`use_javier_recurrent_vjp`, direct CSR, compact pair projection and the direct
state RNN loop. For batches below 32, set both `use_packed_sm120_backward` and
`use_packed_sm120_external_backward` to false: those separate public options
remain batch32-only. Do not weaken their validation to force them on.

The portable preview's rebuilt fused accumulation supports SM61+ and retains
its four-basis requirements. Automatic native selection admits SM75 ordinary
FP16 temporal backward and existing SM86+ paths; other older architectures
remain explicit opt-ins while qualification is pending. The
FP32 temporal-carry reference remains available independently of the default
FP16 temporal-gradient profile.

## Source attribution

Credit: Javier Galvan, `JavierGalvan9/V1_GLIF_model`.

- The variable-batch tile, scaled projection and sparse weight-update design
  adapts `v1_model_utils/cuda_csr_recurrent/csr_recurrent_ops.cu.cc` and
  `event_weight_grad.cuh` at `2c52ec10c1eee409ddf900f8a5b8460cf9c46d24`.
  DPointNet retains its canonical edge ordering, compute-dtype basis and
  TensorFlow accumulator ownership. The general path retains positive-only
  weight-gradient events; the established scaled Javier path retains its nonzero
  event convention, with signed-activity tests. Physical spike counts are nonnegative.
- The opt-in device Poisson algorithm and static-bound/pipelined direct loop
  are based on `v1_model_utils/models.py` at
  `82755f21d2389b667aee17261f6e01db0f7e3ee1`. DPointNet already had a single
  loop bound, so the redundant-bound removal itself needed no port.

## Device Poisson sampling

`use_device_poisson` defaults to false. When explicitly enabled, it generates
stateless float64 uniforms and maps them through a precomputed Poisson CDF
using `tf.searchsorted`. SciPy builds the CDF through the last float64-resolvable
quantile, with the remaining sub-resolution tail assigned to the final count.
The rate is rounded to the selected compute dtype as in the original sampler.
This is Poisson sampling, not Bernoulli sampling. It supports multiple events
per timestep and preserves replica, rollout and timestep seed separation.

The algorithm changes seeded realizations relative to
`tf.random.stateless_poisson`; distributional equivalence is not bitwise
equivalence or training-quality equivalence. Record the option in experiment
provenance, and qualify exact checkpoint replay before a new workload.

## Validation and limits

Changing batch size or parallel/series condition layout changes the training
workload, not just the execution speed. Compare identical update semantics and
record the effective model batch. A speed comparison that enables device Poisson
is not an unchanged-seeded-input control or evidence of equivalent learning.

### Matched RTX8000 batch16/32 execution check

At executable revision `9dcd65614bf2fe5b144c53e2abc7fa35c2b12ed6`, the
automatic profile passed full 66,658-neuron V1+LGN fresh/trained runs at both
batch16(12evoked+4spontaneous) and batch32(24+8) in one RTX8000 allocation.
Each case applied23updates:3excluded warmups and20synchronized measured
input-fetch+training+optimizer+rules+shadow-refresh updates. Starting masters
matched between batches; trained cases strictly restored the same epoch64
checkpoint and regenerated batch-sized initial state.

| State | Batch | Median update | Samples/s (mean time) | TF peak |
| --- | ---: | ---: | ---: | ---: |
| Fresh | 16 | 2.891s | 5.525 | 8.10GiB |
| Fresh | 32 | 4.989s | 6.405 | 13.07GiB |
| Trained epoch64 | 16 | 3.020s | 5.278 | 8.16GiB |
| Trained epoch64 | 32 | 5.854s | 5.434 | 13.48GiB |

Batch16 takes less time per update; batch32 processes twice as many samples,
giving15.9% higher fresh throughput and3.0% higher trained throughput in this
bounded comparison. Native accumulation executed at both batches; packed
backward executed only at32. Strict checkpoint roundtrips, finite constrained
masters and refreshed shadows passed with no rejections or initialization
fallbacks. TF peaks include all23updates, not total driver VRAM.

This used the selected experimental NEST soft-reset V1 qualification recipe,
ordinary FP16 temporal gradients, FP32 masters/optimizer slots, unchanged
losses/ExpAdam/LGN/raw Poisson and default BFC memory growth. It is not a
training recommendation, convergence result or cross-batch update-equivalence
claim. There is no matched old-code RTX8000 batch16 run here to isolate the
causal contribution of an individual optimization.

Tests cover non-power-of-two batches, silent-neuron spike adjoints, canonical
gradient ordering and checkpoint replay. GPU and topology restrictions still
apply; successful execution at one batch size does not qualify another device.
Rebuild CUDA operators for the selected source and hardware before adoption.
See [performance configuration](dpointnet_parity.md) for precision and benchmark
guidance, and [startup preprocessing](dpointnet_startup.md) for developer details.