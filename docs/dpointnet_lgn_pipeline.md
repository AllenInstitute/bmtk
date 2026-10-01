# Per-device LGN input generation

This opt-in pipeline adapts Javier Galvan's
`stim_dataset.py:DriftingGratingLGN` and parameter-only input pipeline from
`JavierGalvan9/V1_GLIF_model` at
`2c52ec10c1eee409ddf900f8a5b8460cf9c46d24`. It concerns input placement and
prefetching, not a new sensory model. This guide is for users configuring LGN
inputs on local GPU strategies.

## Configuration

Add `"use_device_generation": true` at the `lgn_tf` input-module level:

```json
{
  "lgn_evoked": {
    "node_set": "lgn",
    "input": "spikes",
    "module": "lgn_tf",
    "use_device_generation": true,
    "stimulus_type": "drifting_gratings",
    "stimulus_options": {
      "row_size": 80,
      "col_size": 120,
      "temporal_f": 2.0
    }
  }
}
```

The setting defaults to false. It requires drifting gratings, an explicit
stimulus seed or run default seed, and a local single-worker strategy. Existing
fixed/list orientations, regular orientations, phase, rotation, contrast and
pre/post delays retain their semantics. Direct `create_generator` calls remain
the host-reference API; `DataIterator` selects the per-device batch path.

Initial-state warmup always uses the host-reference iterator, even when its LGN
module enables device generation. Warmup runs outside `strategy.run` and needs a
regular tensor rather than replica-local batches. This preserves its seeded
inputs and recovery behavior without changing device generation for training
or inference.

The host dataset yields only orientation, phase and spike seeds. Each local
replica owns LGN tensor copies and generates its spikes on its own device.
Concatenation also happens on the consuming device. Multiple replicas generate
concurrently with one worker per local device. The data iterator prefetches
one batch in one additional worker; `close()` cancels pending work and joins all workers.
`prefetch_device_inputs=False` on `DataIterator` disables this prefetch for
diagnostics. Other input generators retain their existing behavior.
Device input fetching must stay eager; wrapping `next_spikes()` in `tf.function`
raises explicitly so a traced graph cannot capture and reuse a stale batch.

## Seeded sampling

DPointNet's existing training iterator broadcasts a batch across replicas. This
pipeline preserves that sample assignment: every replica generates the same seeded
batch locally. It does not silently switch to Javier's sharded global-batch
assignment or independently reseed replicas. Different distributed sampling
would be a separate scientific/protocol choice.

The outer batch loop remains eager while the existing spatial/temporal filter
functions remain compiled to preserve existing seeded spikes. This retains existing Poisson/BKG
behavior; LGN spike sampling retains the existing Bernoulli-from-rate rule.

## Validation and limits

Single-GPU checks on RTX 3090 compared real LGN batches against the host path
with exact spikes and stimulus signatures, and exercised native optimizer updates
with fresh generated batches. Placement and prefetching are not a guarantee of
speedup; measure end-to-end input and update time on your workload.

Two logical CPU replicas verify local tensor placement and broadcast-compatible
seeded outputs. Physical multi-GPU speed and topology-dependent transfer savings
have not been measured; this is not multi-GPU hardware qualification.