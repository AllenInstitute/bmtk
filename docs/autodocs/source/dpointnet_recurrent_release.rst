.. _dpointnet_recurrent_release:

Fixed recurrent short-term plasticity
=====================================

Use this page to give recurrent synapses fixed, type-specific short-term
plasticity (release dynamics) during training and inference. The release
parameters are fixed inputs taken from physiology; the trained quantities are
still the recurrent and input weights.

Model
-----

Each recurrent edge ``source -> target`` whose (source class, target class)
pair has release parameters transmits the normalized expected response of a
facilitating and depressing synapse. With facilitation ``f`` and availability
``a`` (resting values ``f = 0`` and ``a = 1``), a presynaptic arrival transmits
``w * a * p / p0`` where ``p = p0 + (1 - p0) * f``. After the arrival,
``f += (1 - f) * facilitation_amount`` and ``a`` is depleted by ``p``
(``depression_amount = -1``) or by ``depression_amount``. Between arrivals ``f``
decays with ``facilitation_tau_ms`` and ``a`` recovers with
``depression_tau_ms``. With ``p0 = 1`` and no facilitation or depression the
edge is static, and the cell reproduces ``GLIF3Cell``.

The state depends only on the presynaptic spike train and the parameters, so
it is kept once per (source neuron, parameter group) and each edge reads it at
its own delay. Edges of unsupported pairs and neurons without a class stay
static.

Configuration
-------------

Select the cell and pass the release table in ``rnn_cell_params``:

.. code-block:: json

   {
     "rnn_cell_params": {
       "cell_model": "glif3_release",
       "dynamics_mode": "nest",
       "recurrent_release": {
         "node_classes": "release/node_classes.npy",
         "pair_parameters": "release/pair_parameters.npy",
         "pair_supported": "release/pair_supported.npy"
       }
     }
   }

Each entry is an array or a ``.npy`` path:

``node_classes``
   Integer class of every V1 neuron in model order, shape ``[n_nodes]``;
   ``-1`` means no release dynamics. All neurons of one node type must share a
   class.
``pair_parameters``
   Float table of shape ``[classes * classes, 5]``; row ``pre * classes + post``
   holds ``base_release_probability``, ``facilitation_amount``,
   ``depression_amount``, ``depression_tau_ms`` and ``facilitation_tau_ms``.
``pair_supported``
   Boolean ``[classes * classes]``; unsupported pairs stay static.

NEST dynamics are required. The cell adds two float32 states of shape
``[batch, keys]`` after the ``GLIF3Cell`` states.

Training route
--------------

Training is supported only on the direct recorded adjoint, which the automatic
route selects for this cell. Gradients are exact there: they include credit to
future release state and to silent connections, through the event-gated
surrogate of the binary forward. The ordinary cell path is used for warm-up,
validation and inference. Its forward is exact, but a tape through it is not,
so ``TrainingEngine`` raises if a training update comes from any other route.
Check the ``Training route:`` log line or ``training_route_report`` as usual.
``recurrent_spike_gradient_scale`` scales the release-correction credit in the
same way as the static recurrent credit.

Weights, checkpoints and exports are unchanged: every recurrent edge keeps one
weight in the recurrent master. Internally, release edges also feed a
correction input whose weights are a tied copy of the recurrent master. Its
gradient is folded into the master, the optimizer does not update it (no
slots, no weight decay), and the cell re-ties it after every update or
assignment.

Cost
----

On an RTX 3090 with the 66,658-neuron V1 network, batch 32, mixed precision and
about 314,000 release keys (80% of recurrent edges), one forward and backward
step takes 17.5 ms versus 3.3 ms for the static cell, and training peaks at
about 17-20 GiB of TensorFlow memory versus about 14 GiB static.
