# Variable-batch parity follow-up

This follow-up extends `1f0e84c2396f06140c74c8530ac5fc8276581f0a` on
`feature/dpointnet-javier-parity`; the older pin does not include these changes.
Do not change an existing experiment's source or random sampler in place.

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

The carrier wrapper now accepts forward acceleration options. The cell uses
them for variable batches; batch32 preserves its established carrier-forward
dispatch. Retain `use_fused_recurrent_accumulation`,
`use_javier_recurrent_vjp`, direct CSR, compact pair projection and the direct
state RNN loop. For batches below 32, set both `use_packed_sm120_backward` and
`use_packed_sm120_external_backward` to false: those separate public options
remain batch32-only. Do not weaken their validation to force them on.

Fused accumulation retains its SM86+ hardware and four-basis requirements. The
FP32 temporal-carry reference remains available independently of the default
FP16 temporal-gradient profile.

## Newer upstream adaptations

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

Not imported wholesale: Javier's new OSI/DSI loss formulation, shared
gray-screen initialization policy, multi-GPU validation aggregation and plotting
worker. Those change scientific or application-level behavior and are outside
the native-update speed qualification. DPointNet's canonical master ordering
also avoids Javier's checkpoint-time runtime-edge permutation.

## Evidence

Fresh3+20forward/reverse-order pairs on RTX3090 measured1.137-1.156s at batch8
(Javier1.143-1.148s) and1.646-1.652s at batch16 (Javier1.595-1.619s).
Batch13 measured1.492s and original-sampler batch32 measured2.589s.
The small-batch speed recipe explicitly enables `use_device_poisson`; it is not
the unchanged-RNG control and is not a trained-network performance claim.

The follow-up report and retained measurements are in
`/local2/results/dpointnet_rule_search/parity_followup_20260930/REPORT.md`.
Use its final qualification status, not intermediate screening results, when
selecting a source for a new experiment.

The branch also includes [startup preprocessing](dpointnet_startup.md).
Pin the selected Git revision in each new project's execution contract and
rebuild CUDA operators for its hardware; fetching the branch does not install
or relink it into any existing environment.