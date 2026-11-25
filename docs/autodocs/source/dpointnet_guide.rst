#########
DPointNet
#########

DPointNet simulates spiking neuronal networks in TensorFlow and can train their
synaptic weights against a chosen objective. It reads network files in SONATA,
the same format used by other BMTK simulators. You can use it for simulation
without training, or for learning in a recurrent network.

The supported neuron model is **GLIF3**: a generalized leaky integrate-and-fire
model with two after-spike currents (ASC), which describe adaptation after a
spike. The Python class is named ``GLIF3Cell``. One instance represents the
network's neuron population, not a single biological neuron.

During training, a smooth **surrogate derivative** replaces the derivative of
the spike threshold. **Backpropagation through time** (BPTT) then assigns later
errors to earlier network activity. The chosen loss functions define which
observables the network is trained to fit.

For the scientific background, see
`Ito et al., 2026 <https://www.biorxiv.org/content/10.64898/2026.03.13.711751v1>`_.
Use documentation from the same BMTK revision as your installation.

Installation
============

Follow :doc:`installation` for BMTK's base dependencies. DPointNet additionally
requires TensorFlow. The CPU walkthrough below was checked with Python 3.13
and TensorFlow 2.21; TensorFlow 2.13 with Python 3.8 is also covered by
compatibility tests. These are tested combinations, not an exhaustive support
matrix.

From a BMTK source checkout, in an activated environment:

.. code-block:: bash

   python -m pip install -e .
   python -m pip install "tensorflow==2.21.0"

For a Linux NVIDIA GPU setup, TensorFlow's CUDA dependencies can instead be
installed with ``python -m pip install 'tensorflow[and-cuda]==2.21.0'``.
Check the driver and platform requirements in the
`TensorFlow installation guide <https://www.tensorflow.org/install/pip>`_.

**Custom CUDA operators are optional for the first example.** Without them,
automatic acceleration uses compatible TensorFlow paths. Building the operators
also requires a CUDA toolkit with ``nvcc`` and a compatible C++ compiler; that
is a separate step in :doc:`dpointnet_performance`.

Run a small simulation
======================

The source checkout includes a complete 300-neuron example in
``examples/dpointnet_all2all``: 200 excitatory neurons, 100 inhibitory neurons,
100 virtual input nodes, component files and a prebuilt SONATA network.
There is no network download or build step.

Copy that directory to a writable work area so results do not modify the
original example. On Linux or macOS, from the checkout root:

.. code-block:: bash

   mkdir -p "$HOME/bmtk-examples"
   cp -R examples/dpointnet_all2all "$HOME/bmtk-examples/"
   cd "$HOME/bmtk-examples/dpointnet_all2all"
   CUDA_VISIBLE_DEVICES="" python run_dpointnet.py config.inference.json

This runs 500 steps at 1 ms per step, with batch size one, on the CPU. The
example explicitly uses FP32 and legacy dynamics to preserve its original
behavior; those settings are not the omitted defaults for a new GLIF model.
Its random input uses a Bernoulli spike generator, not a Poisson generator.
See :ref:`dpointnet-background-inputs` before using it as background (BKG) drive.

The script displays a spike raster and writes ``output/spikes.h5``,
``output/voltages.h5`` and ``output/log.txt``. Successful completion is reported
as ``RNN.run() completed.`` The raster is an activity check, not evidence that
the network reproduces a biological target. This small example needs no GPU;
its resource use is not an estimate for a cortex-scale network.

On a headless machine, set ``MPLBACKEND=Agg`` before running the command. To
save a plot instead of displaying one, use the same configuration from Python:

.. code-block:: python

   from bmtk.simulator import dpointnet

   config = dpointnet.Config.from_json("config.inference.json")
   config.build_env()
   rnn = dpointnet.RNN.from_config(config)
   results = rnn.run()
   results.spikes.raster(batch_nums=0, show=False)

   import matplotlib.pyplot as plt
   plt.savefig("raster.png")
   rnn.cleanup()

Keep the example directory as the working directory: its file paths are
relative to that location. ``config.build_env()`` prepares output directories
and logging. ``RNN.from_config()`` loads the network and execution settings;
``run()`` builds the model, trains it if configured, then runs and saves the
configured inference.

Understand the configuration
============================

Start by editing a complete example, rather than assembling an incomplete
fragment. The important top-level fields are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - JSON location
     - Purpose
   * - ``manifest``
     - Path variables such as ``$NETWORK_DIR`` and ``$OUTPUT_DIR``.
   * - ``run``
     - ``dt`` in milliseconds, ``seq_len`` in steps, batch size, seed and precision.
   * - ``rnn_cell_params``
     - ``cell_model: "GLIF3Cell"``, neuron dynamics, learning derivatives and acceleration.
   * - ``components`` and ``networks``
     - Model-component directories and SONATA node/edge files.
   * - ``inputs``
     - Named input generators for virtual input populations.
   * - ``initial_state``
     - Optional shared initial-state generator.
   * - ``training``
     - Training duration, optimizer, learning rate, conditions, losses and callbacks.
   * - ``inference``
     - Input names and initial state for simulation; may also specify its own output.
   * - ``output``
     - Default result directory, logging, spike and voltage filenames.

For example, ``seq_len=500`` and ``dt=1.0`` describe 500 ms. A batch contains
multiple input trials processed together; it does not extend the time window.
Choose both explicitly for your experiment.

The following is a **JSON fragment**, to merge into a complete configuration:

.. code-block:: json

   {
     "run": {
       "seq_len": 500,
       "dt": 1.0,
       "batch_size": 1,
       "default_seed": 3000,
       "precision": {
         "mixed_precision": false
       }
     },
     "rnn_cell_params": {
       "cell_model": "GLIF3Cell"
     }
   }

Here FP32 is selected explicitly for a simple CPU starting point. For GPU
training, :doc:`dpointnet_training_standards` explains the mixed-precision
defaults and the switches for wider state and gradients.

Use ``inference.initial_state`` or ``training.initial_state`` for a
stage-specific state generator. ``{"module": "zero_state"}`` starts from zero;
``from_input`` uses a preparatory input rollout. Do not substitute one policy
for another when comparing experiments.

Import your own network
-----------------------

Use the :doc:`BMTK Network Builder <builder>` or existing SONATA files.
Recurrent neurons need GLIF component JSONs, while edge types identify their
synaptic dynamics and weights. Virtual input populations also need their
edges into the recurrent population.

The example's ``components`` and ``networks`` sections show the complete file
layout. For raw synaptic kinetics without precomputed basis coefficients, see
:doc:`dpointnet_alpha_basis`. For visual input generated by a lateral
geniculate nucleus (LGN) model, see :doc:`dpointnet_lgn_pipeline`.

.. _dpointnet-background-inputs:

Background inputs (BKG)
----------------------------------------

Background drive has two independent choices: **which distribution generates
events**, and **where generation happens**. Internal generation is not a third
distribution: ``poisson_spikes_internal`` generates Poisson counts inside
``GLIF3Cell`` instead of supplying a spike sequence from an external generator.
Here, "external" means outside the cell, not a separate simulator.

.. list-table::
   :header-rows: 1
   :widths: 28 20 24 28

   * - ``inputs.<name>.module``
     - Distribution
     - Generated where
     - Events per source per step
   * - ``poisson_spikes``
     - Poisson
     - External NumPy generator
     - Counts: 0, 1, 2, ...
   * - ``bernoulli_spikes``
     - Bernoulli
     - External NumPy generator
     - Binary: 0 or 1
   * - ``random``
     - Bernoulli
     - Alias of ``bernoulli_spikes``
     - Binary: 0 or 1; not Poisson
   * - ``poisson_spikes_internal``
     - Poisson
     - Inside ``GLIF3Cell``
     - Counts: 0, 1, 2, ...
   * - ``noisy_current``
     - Poisson
     - Alias of ``poisson_spikes_internal``
     - Counts projected to synaptic current; not Gaussian noise

All five modules are selected with ``"input": "spikes"`` in JSON, including
``noisy_current``. Both Poisson routes use the input population's connectivity
and synaptic weights to convert counts into current. Internal generation does
not remove the need for virtual input nodes and their edges.

Distribution and units
~~~~~~~~~~~~~~~~~~~~~~

``firing_rate`` is in Hz **per virtual source**, not the resulting firing rate
of a recurrent neuron. For a generator timestep ``dt`` in milliseconds, the
mean count is ``lambda = firing_rate * dt / 1000``. Poisson sampling draws an
integer count with that mean. Bernoulli sampling instead compares a uniform
draw with ``p = firing_rate * dt / 1000``; use only ``0 <= p <= 1``.

These are not interchangeable models. At 250 Hz and 1 ms, both have mean
0.25 events per step, but Poisson permits multiple events (about 2.65% of
steps have two or more); Bernoulli never does. Their variances also differ.
For a model specified to have Poisson BKG, choose a Poisson module and retain
counts when storing, casting or replaying inputs; do not convert them to
Boolean spikes. The all-to-all example's Bernoulli input is a demonstration,
not the Poisson BKG recipe. LGN visual sampling is a separate input policy;
see :doc:`dpointnet_lgn_pipeline`.

Configure external Poisson BKG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is a **JSON fragment** for an existing network with a virtual
``bkg`` population and edges from it to the recurrent neurons. Merge it into
the complete configuration; replace ``bkg`` with your population name.

.. code-block:: json

   {
     "run": {"dt": 1.0},
     "inputs": {
       "background": {
         "node_set": "bkg",
         "input": "spikes",
         "module": "poisson_spikes",
         "firing_rate": 250.0,
         "seed": 3000
       }
     },
     "inference": {"inputs": ["background"]}
   }

The input name ``background`` is arbitrary; ``node_set`` identifies the loaded
input population. Include the input name in ``training.parameters[].inputs`` too when
training, alongside any other inputs required by that condition.
``poisson_spikes`` also accepts a nonempty list of rates: it chooses one
uniformly for each generated trial, shared across that trial's sources.
This is not a per-source rate vector. Set the rate explicitly rather than
relying on the Poisson default of 250 Hz.

For intentional Bernoulli drive, replace ``module`` with ``bernoulli_spikes``
and supply a scalar ``firing_rate``. Its seed behavior differs; see below.

.. warning::

   In this revision, the standard data iterator does not pass ``run.dt`` to
   external generators. ``poisson_spikes.create_generator(dt=...)`` defaults
   to 1 ms, and a JSON input-level ``dt`` does not override it.
   The fragment above therefore explicitly uses ``run.dt=1.0``.
   For another timestep, use internal Poisson or a custom external-input path
   that passes the matching timestep. Bernoulli also defaults to 1 ms, but
   supports an input-level ``dt`` override; set it to match ``run.dt``.

Configure internal Poisson BKG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the external background module with ``poisson_spikes_internal``.
The cell samples counts for every batch member and input source at each
timestep, using ``run.dt``, then applies the input projection. There is no
external BKG spike tensor to generate or concatenate.

This **JSON fragment** assumes ``visual`` is an already-defined external input
for another population, such as LGN; retain its existing definition:

.. code-block:: json

   {
     "rnn_cell_params": {"noise_seed": 3000},
     "inputs": {
       "background": {
         "node_set": "bkg",
         "input": "spikes",
         "module": "poisson_spikes_internal",
         "firing_rate": 250.0
       }
     },
     "inference": {"inputs": ["visual", "background"]}
   }

Keep ``background`` in the stage's input list even though its generator
returns no external sequence. The standard data iterator requires at least
one active external spike generator: a stage containing only internal BKG
raises ``DataIterator did not receive any active spike generators.`` Internal
BKG alone is therefore not a drop-in replacement for the all-to-all example's
sole external input. Use scalar ``firing_rate`` for internal BKG.

Seeds, replay and execution placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The seed controls are module-specific; do not assume ``run.default_seed``
configures every background generator:

* External ``poisson_spikes`` uses its input-level ``seed`` with NumPy's
  ``default_rng``. Omission uses an unseeded generator. A new
  ``create_generator()`` call starts a new RNG from that seed.
* ``bernoulli_spikes`` and its ``random`` alias currently use a fixed NumPy
  seed of 3000 at module construction. An input-level ``seed`` does not
  override it.
* Internal Poisson uses ``rnn_cell_params.noise_seed`` (default 0), the
  logical-rollout stream and timestep counter. Standard training/inference
  advances the stream between rollouts and preserves draws during checkpoint
  recomputation. Distributed execution also folds in the replica/rank identity.

External and internal Poisson use different RNG implementations and stream
lifecycles: matching rates and numeric seeds does **not** give identical event
sequences. Record the module, rate, timestep, seed controls and source revision
when reproducing an experiment.

Internal generation is not necessarily GPU generation. By default,
``rnn_cell_params.use_device_poisson`` is false and counts are sampled on CPU.
Automatic ``use_prepared_cpu_poisson`` may prepare the same counts for a
rollout in advance. Explicit ``use_device_poisson=true`` selects a device
Poisson sampler but changes seeded realizations; it does not select Bernoulli
sampling. See :doc:`dpointnet_performance` for execution options.

Train the example
=================

For this small legacy example, disable activation checkpointing explicitly.
Its integer refractory initial state is incompatible with the checkpointed
legacy loop in this revision. This is an example-specific limitation, not a
change to the general checkpointing default, and does not disable BPTT.

From the same example directory:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" python - <<'PY'
   from bmtk.simulator import dpointnet

   config = dpointnet.Config.from_json("config.train.all.json")
   config["training"]["gradient_checkpointing"] = False
   config.build_env()
   rnn = dpointnet.RNN.from_config(config)
   rnn.run()
   rnn.cleanup()
   PY

This complete configuration runs two epochs of three steps each, fits a
synthetic 15-Hz target for all neurons, and then runs inference.
The default callbacks write ``callbacks_outputs/losses.csv`` and export
trained edge files under ``callbacks_outputs/trained_weights/``.
This is a workflow demonstration, not a recommended learning rate or
biological target for another network.

SONATA HDF5 exports use lossless gzip compression by default. For spike and
voltage results, select ``compression`` in the ``output`` section (or an
inference's own ``output`` section):

.. code-block:: json

   {
     "output": {
       "spikes_file": "spikes.h5",
       "voltages_file": "voltages.h5",
       "compression": "gzip"
     }
   }

Choose ``"gzip"``, ``"lzf"``, an integer gzip level from 1 to 9, or
``"none"``/JSON ``null`` to disable compression. This includes split-batch
files and voltage mapping datasets; CSV output is unchanged.
Programmatic ``results.spikes.to_sonata(...)``,
``results.voltages.to_sonata(...)`` and ``results.save_results(...)`` accept the
same ``compression`` option (Python ``None`` disables it).
For trained edge exports, use ``training.callbacks.sonata_compression``;
see :doc:`dpointnet_training`.

Continue with :doc:`dpointnet_training` for ordinary training choices,
validation, saved weights and optional regularizers. For your own objectives
or training hooks, follow :doc:`dpointnet_custom_training`; no BMTK edits are needed.

Multi-GPU support
=================

See :doc:`dpointnet_multi_gpu` for sample-parallel execution across multiple
GPUs, including configuration, launch commands, a runnable example and
supported restrictions. Each GPU holds the full network; the sample batch
is divided between GPUs.
Start with the explicitly enabled MPI approach for the qualified workload;
single-process MirroredStrategy is an opt-in alternative. This recommendation
does not change execution when ``run.distribution`` is omitted.

Dynamics and simulation
=======================

Omitting ``rnn_cell_params.dynamics_mode`` selects ``"nest"``, DPointNet's
**NEST-compatible dynamics**. This implements NEST-compatible timing,
adaptation and delay handling in TensorFlow; it does **not** run the NEST
simulator or require a NEST installation. The mode supports both training
and simulation.

Explicit ``"legacy"`` remains available
for reproducing an older model. Changing dynamics changes the simulated
trajectory; it is not just a speed setting.

New NEST GLIF configurations use **hard reset forward, soft-surrogate backward**.
Voltage resets and refractory clamps stay hard during both training and inference;
the approximate learning derivative retains continuous voltage credit.
See :doc:`dpointnet_hard_reset_training` for the rule and its limitations.
Explicit ``hard_reset=false`` retains soft reset. Explicit ``hard_reset=true``
without a gradient mode retains the historical exact-clamp derivative; for
hard-forward training set ``hard_reset_gradient_mode="soft_surrogate"``.
Simulation of learned weights should retain the training
reset policy. To study hard-reset dynamics, build a separate inference-only
model with that setting and assess the changed behavior.

NEST-compatible mode timestamps spikes at the end of the timestep. Its ``dt`` must be a
positive multiple of 0.001 ms, and external delays must be at least one step.
Compatibility describes the implemented dynamics; numerical precision and
fitted synaptic waveforms can still produce differences from a NEST simulation.

For a runnable comparison, see
:download:`the 300-neuron reset and precision example README
<../../../examples/dpointnet_nest_comparison/README.md>` in the repository's
``examples/dpointnet_nest_comparison/`` directory.
It holds the network and replayed input fixed, compares FP32 and mixed
precision under both reset policies, and includes an actual NEST reference
through PointNet. Mixed/soft illustrates the standard forward settings
without changing weights through training.

For simulation only, omit ``training`` and retain ``inference``. For manual
execution, use ``rnn.build()``, ``rnn.train()`` and ``rnn.run_inference()``;
unlike ``run()``, ``run_inference()`` returns results without automatically
saving them. Long simulations need complete state, delay history and random
stream handling between chunks; repeated independent calls are not an
automatic continuous simulation.

Next steps
==========

.. toctree::
   :maxdepth: 2
   :caption: Multi-GPU support

   Sample-parallel execution <dpointnet_multi_gpu>

.. toctree::
   :maxdepth: 1
   :caption: Training and configuration

   dpointnet_training
   dpointnet_custom_training
   dpointnet_training_standards
   dpointnet_hard_reset_training

.. toctree::
   :maxdepth: 1
   :caption: Optional network and input features

   dpointnet_alpha_basis
   dpointnet_lgn_pipeline
   dpointnet_recurrent_release

.. toctree::
   :maxdepth: 1
   :caption: Performance and troubleshooting

   dpointnet_performance
   dpointnet_input_recovery
   dpointnet_pascal

For implementation details and custom runners, use
:doc:`dpointnet_development` in the developer guide.
