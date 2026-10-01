# Pascal GPU compatibility

This guide is for users of Pascal GPUs and developers maintaining older-device
compatibility. DPointNet includes Pascal-compatible CUDA atomics/warp grouping
and GPU LGN spatial filtering without changing neuronal dynamics, losses,
configured precision or background distribution.

**General NEST execution on Pascal is not qualified.** A full GTX 1080 Ti
regression run failed, including fused NEST launch-resource errors. The legacy
batch-16 and LGN smoke results below are narrower checks, not a complete Pascal
support claim.

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
are met. Do not switch an existing experiment's precision or dynamics to work
around compatibility failures. Legacy remains the default; NEST training is
experimental and is not recommended on any hardware.

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

## Validation scope and limits

| Surface | Scope |
|---|---|
| CUDA currents and gradients | Focused independent-reference tests passed on GTX 1080 Ti / SM61. |
| LGN filters | Real spatial and temporal shapes passed CPU-reference checks with GPU placement required and a `1e-6` tolerance. |
| Legacy training | A three-update V1 batch-16 smoke passed; long-run training and higher-activity memory capacity are not qualified. |
| General NEST execution | Not qualified: the full Pascal suite failed, including fused NEST launch-resource errors. |
| Titan Xp and physical multi-GPU | Not separately measured. |

The legacy smoke used TensorFlow 2.21, CUDA 12.9 and cuDNN 9.25 on GTX 1080 Ti:
66,658 neurons, 500-ms sequences, effective batch 16 (12 evoked + 4 spontaneous),
selective forward state, ordinary FP16 temporal gradients, checkpoint chunk 25,
original Poisson BKG, default BFC and fresh prefetched LGN inputs. TensorFlow
allocation peaked around 7.9 GiB and driver reservation around 10.4 GiB on an
11-GiB card, leaving limited headroom. This does not establish safe memory for
trained higher-activity networks, longer sequences, other losses or long runs.
There is no qualified steady-state performance claim.

Independent tests retain FP32 value/input-gradient coverage. Nonlinear
chunk-boundary tests also use an FP64 synthetic oracle to isolate indexing from
FP32 reduction cancellation; production precision and tolerances are unchanged.
Passing the complete SM86 suite is not a complete Pascal regression pass.

Rebuild operators for your source and GPU, then smoke-test the exact workload
before adoption. Do not interpret a successful build or a focused LGN test as
qualification of every execution path.