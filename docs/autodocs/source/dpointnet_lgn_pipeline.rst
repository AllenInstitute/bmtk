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

Native grating operators
~~~~~~~~~~~~~~~~~~~~~~~~

Device generation of drifting gratings uses the native exact-factorization
operators (``custom_ops/_lgn_grating_ops.so``, built by ``custom_ops/build.sh``
beside the state and projection libraries) whenever a GPU and the built
library are available. A grating frame is ``Im(e^{i(k.r + w f + phi)})`` and
every stage before rectification is linear, so each unit's filtered response is
``Im(A R(t))`` with a per-sample complex spatial coefficient ``A`` (computed in
float64 on the device) and a per-unit complex temporal response ``R``
precomputed once in float64. The resulting rates equal the movie -> spatial ->
temporal TensorFlow path to float32 rounding, and spikes use the same stateless
float32 uniforms, so the stimulus protocol, seeds and trial assignment are
unchanged. On the 17,400-unit LGN at batch 32 and 500 steps an RTX 3090
generates a batch in about 0.11 s instead of 0.88 s.

The stimulus option ``lgn_backend`` selects the implementation: ``"auto"``
(default), ``"native"`` (raises when the operators or a compatible GPU are
missing) or ``"tensorflow"``. ``native_chunk`` (1-16, default 16) sets how many
samples' uniforms are alive at once (about 33 MiB per sample at 17,400 units
and 500 steps). ``DriftingGratingLGN.backend`` reports the resolved choice.
The operators contain no architecture-specific instructions and are part of
the fat binary (SM 6.1 upward). Structure adapted from Javier Galvan's
``v1_model_utils/cuda_lgn`` (credit: Javier).

Native movie operators
~~~~~~~~~~~~~~~~~~~~~~

Arbitrary movies (natural movies, flashes, gray screens, or gratings outside
the sinusoidal factorization) go through ``LGN.firing_rates(movie)``, which
evaluates the same spatial-filter -> bilinear-sample -> temporal-filter ->
rectification pipeline with the native movie operators
(``custom_ops/_lgn_movie_ops.so``, ``DpointnetLgnMovieResponse`` /
``DpointnetLgnMovieSpikes``) when a GPU and the built library are available,
and otherwise with ``spatial_response`` + ``firing_rates_from_spatial``. The
two agree to float32 rounding. ``LGN(..., movie_backend="auto"|"native"|
"tensorflow")`` selects the route; ``LGN.movie_backend`` reports it. ``auto``
keeps TensorFlow for configurations outside the operators' limits (frames above
1024 rows x 11-column groups, for example 120 x 120, or spatial filters wider
than 27 taps; ``lgn_movie_ops.movie_support_error`` names the reason, which an
explicit ``"native"`` request raises). The
movie may be ``[time, rows, cols, 1]`` or a batch ``[batch, time, rows,
cols(, 1)]`` in float16 or float32. ``custom_ops.lgn_movie_ops.MovieLGNKernel``
also samples seeded spikes in chunks (``chunk`` 1-16 samples at a time, or a
``movie(begin, end)`` callable so only one chunk of movie is alive). Gray-screen
probabilities and the TensorFlow fallback of the grating generator use this
route automatically. The spatial pass stages a frame in shared memory and, on
devices whose opt-in shared memory holds no whole frame (V100, Pascal for 80 x
120), filters it in bands of rows; the result is bitwise the same either way
(``DPOINTNET_LGN_MOVIE_MAX_BAND`` caps the band for tests).

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
GPUs with compute capability 7.0 or earlier (including V100) use bounded
GPU spatial filtering and non-XLA firing-rate dispatch to avoid unsupported
cuDNN convolution algorithms. This retains the filter padding, edge
normalization and sampling policy; it does not move LGN generation to the CPU.
GPUs above compute capability 7.0 retain the existing convolution route.
For older GPUs, see :doc:`dpointnet_pascal`; for initial-state generation
interruptions, see :doc:`dpointnet_input_recovery`.
