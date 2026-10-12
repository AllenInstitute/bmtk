.. _dpointnet_lgn_pipeline--per-device-lgn-input-generation:

Visual input from an LGN model
==============================

This optional feature generates spikes from a lateral geniculate nucleus
(LGN) model in response to visual stimuli. It is relevant to visual networks,
not a prerequisite for ordinary DPointNet simulation or training.

You need an LGN input population with its SONATA node/type files and filter
parameters, its edges into the recurrent network, and the matching component
files. Naming an arbitrary virtual population ``lgn`` does not supply those
models. ``examples/dpointnet_v1`` shows the visual-network setup; the small
all-to-all example does not contain LGN assets.

.. _dpointnet_lgn_pipeline--configuration:

Configure a drifting grating
----------------------------

Merge this **JSON fragment** into a complete simulation config whose network
already defines the LGN population:

.. code-block:: json

   {
     "run": {
       "default_seed": 3000
     },
     "inputs": {
       "lgn_evoked": {
         "node_set": "lgn",
         "input": "spikes",
         "module": "lgn_tf",
         "use_device_generation": true,
         "stimulus_type": "drifting_gratings",
         "stimulus_options": {
           "row_size": 80,
           "col_size": 120,
           "temporal_f": 2.0,
           "seed": 42
         }
       }
     }
   }

``row_size`` and ``col_size`` are the stimulus image dimensions.
``temporal_f`` is in Hz. Other stimulus options control orientation, spatial
frequency, phase, contrast and pre/post delays; keep those experimental choices
explicit when comparing runs. Include ``"lgn_evoked"`` in the selected
``inference.inputs`` or ``training.parameters[i].inputs`` list.

Device generation is the default for drifting gratings. It requires a
stimulus seed or ``run.default_seed`` and a local, single-worker TensorFlow
strategy. The high-level RNN uses local ``MirroredStrategy``; programmatic
input iterators also accept a local ``OneDeviceStrategy``.
It is not a multi-worker pipeline.

Set ``use_device_generation=false`` explicitly for unseeded host generation,
unsupported strategies or host firing-rate output. Unsupported device requests
raise instead of changing your input protocol. Gray-screen stimuli use the
host route by default.

.. _dpointnet_lgn_pipeline--seeded-sampling:

Seeds and batches
-----------------

Each local replica generates the same seeded trial batch on its device.
This preserves DPointNet's existing **broadcast** assignment: replicating
the batch does not create new independent trials. It is not **sharding**,
where replicas receive different parts of a global batch.

If a stimulus seed is absent, a configured run seed supplies the generator
seed. Preserve both the seed policy and trial assignment when reproducing
an experiment. This visual-input generator retains its Bernoulli-from-rate
spike sampling; it does not replace the separate Poisson background generator.

Initial-state warmup uses the host-reference iterator even when training
or inference uses device generation. That is an intentional placement
difference, not a new initial-state or sampling policy.

.. _dpointnet_lgn_pipeline--validation-and-limits:

Performance and troubleshooting
-------------------------------

The iterator prefetches one batch and generates inputs on the consuming
devices. Fetching must remain eager: do not put ``next_spikes()`` inside a
``tf.function`` that might capture a stale batch. Close a manually managed
iterator when finished; ``rnn.cleanup()`` closes configured inference inputs.

Seeded host/device comparisons have been checked on a single RTX3090.
Multiple logical replicas are not physical multi-GPU performance validation.
Measure end-to-end input and update time before expecting a speedup.
For older GPUs, see :doc:`dpointnet_pascal`; for initial-state generation
interruptions, see :doc:`dpointnet_input_recovery`.
