DPointNet training
==================

First run the small simulation in :doc:`dpointnet_guide`. Training adds an
objective and an update protocol to the same network. Use
``examples/dpointnet_all2all/config.train.all.json`` as a complete
starting configuration, including its network, input and target files.

Custom objectives and callbacks are regular user extension points, not
changes to BMTK itself. Follow :doc:`dpointnet_custom_training` for complete
Python definitions, registration, JSON configuration and a runnable launcher.

New NEST GLIF configurations default to hard-reset forward dynamics with
approximate soft-reset-style voltage derivatives; see
:doc:`dpointnet_hard_reset_training`. Explicit ``hard_reset=false`` retains
soft-reset training. Existing examples preserve their original reset policies.

Choose an experiment
--------------------

Before training, decide:

* **Target and units:** for example, each neuron's firing rate in Hz.
* **Trainable weights:** recurrent connections, input connections, or both.
* **Input conditions:** named generators and the trials used for fitting.
* **Batch and window:** parallel trials per update and timesteps per trial.
* **Optimizer and learning rate:** chosen for the network and objective.
* **Duration:** ``n_epochs`` and ``steps_per_epoch``; a step executes the
  selected condition-update protocol, not one simulation timestep.
* **Evaluation:** held-out inputs, evaluation frequency and the score used
  to choose saved weights. A lower training loss alone is not generalization.

The following **JSON fragment** replaces ``training`` in the small example.
It uses that example's ``rand_inputs`` generator and a synthetic 15-Hz target:

.. code-block:: json

   {
     "training": {
       "n_epochs": 2,
       "steps_per_epoch": 3,
       "training_approach": "single",
       "gradient_checkpointing": false,
       "learning_rate": 0.001,
       "optimizer": {
         "name": "adam",
         "epsilon": 0.000001
       },
       "initial_state": {
         "module": "zero_state"
       },
       "callbacks": {
         "class": "Callbacks",
         "callbacks_dir": "training_outputs",
         "verbose": "on_step",
         "epoch_store_weights": "latest",
         "sonata_output_dir": "trained_network",
         "losses_table_csv": "losses.csv"
       },
       "parameters": [
         {
           "name": "rate_fit",
           "inputs": ["rand_inputs"],
           "loss_functions": {
             "rates": {
               "module": "TargetFiringRate",
               "firing_rate": 15.0
             },
             "voltage": {
               "module": "VoltageRegularization",
               "voltage_cost": 1.5,
               "penalty_mode": "range"
             }
           }
         }
       ]
     }
   }

The learning rate and coefficients above are illustrative, not defaults or a
convergence recommendation. In the supplied complete example, full voltage
output is explicitly enabled. For compact voltage output in a new config, see
the voltage-regularization section below.

This small legacy example explicitly disables activation checkpointing because
its refractory-state dtype is incompatible with the segmented legacy loop in
this revision. It still uses full BPTT. The general default remains enabled;
this example's exception is not a recommendation for large NEST models.

Each entry in ``training.parameters`` defines a named condition: its inputs,
losses and optional ``batch_size``/``seq_len`` overrides. The condition refers
to input names in top-level ``inputs``. Callbacks belong inside ``training``,
not at the top level.

``training_approach="single"`` uses one condition. With multiple conditions,
``series`` applies an update per condition, ``parallel`` combines conditions
into one model batch, and ``series_accumulate`` accumulates them before one
update. These protocols are not interchangeable: two sequential batch-16
updates differ from one combined batch-32 update.

.. _single-host-data-parallelism-engineering-preview:
.. _one-process-per-gpu-mpi-engineering-preview:

Multi-GPU execution
-------------------

For sample-parallel training, see :doc:`dpointnet_multi_gpu`. That section
explains the execution modes, per-GPU versus global batch sizes, configuration,
MPI launcher, runnable example, output ownership and supported scope.
Multi-GPU sample batches are distinct from the condition-update protocols above.
MPI is the recommended starting approach when explicitly enabling multi-GPU
training; MirroredStrategy remains an opt-in alternative. Neither is enabled
automatically by this recommendation.

Weights and objectives
----------------------

``rnn_cell_params.train_recurrent`` controls recurrent-weight training.
For an input population, set ``inputs.<input_name>.trainable`` explicitly
to true or false. Do not assume all inputs are trainable just because
recurrent training is enabled.
Freezing a weight group and changing its learning rate are different choices.

.. list-table:: Common loss modules
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - ``TargetFiringRate``
     - Fit numeric rates in Hz: a scalar target or a model-ordered vector.
   * - ``SpikeRateDistributionTarget``
     - Fit a population firing-rate distribution instead of individual neuron targets.
   * - ``OrientationSelectivityLoss``
     - Visual-response selectivity objective for appropriately configured stimulus conditions.
   * - ``VoltageRegularization``
     - Penalize voltages outside a range or their distance from threshold.
   * - ``SynchronizationLoss``
     - Penalize the configured population synchrony measure.
   * - ``EMDWeightRegularization``
     - Compare weight distributions; this is not a firing-rate target.

Loss coefficients and selected populations determine the experiment. Inspect
individual loss terms, firing-rate distributions and voltage behavior, not
only their sum. The visual-cortex example in ``examples/dpointnet_v1`` needs
its own model/data preparation and is not the first-run tutorial.

The built-in ``TargetFiringRate`` does not load CSV filenames.
Its ``compiled_spike_counts`` option defaults to ``"auto"`` for the bounded
native GPU route described in :doc:`dpointnet_performance`; explicit booleans
force the compiled FP32 count reduction or retain the ordinary reduction.
This affects loss execution, not the selected target units or optimizer.

The separate ``run_dpointnet.firing_rate_reg.py`` example registers a custom
CSV-aware loss for the individual/grouped CSV configurations. Use that
entrypoint when following those examples; it is not an automatic extension
of the generic JSON loader.

For your own target format or objective, use the loss interface and
registration workflow in :doc:`dpointnet_custom_training`.

``training.optimizer.name`` accepts ``adam``, ``exp_adam`` or ``sgd``.
Adam uses additive updates; ``exp_adam`` uses exponentiated updates, so the
same learning rate is not necessarily comparable. A numeric
``training.learning_rate`` is constant; a dictionary selects a registered
schedule, as in the supplied example. There is no universally suitable
optimizer or learning rate.

Optional optimizer clipping accepts at most one of ``clipnorm``,
``clipvalue`` and ``global_clipnorm``, each finite and positive. Clipping is
disabled when omitted. It acts on unscaled gradients and does not repair
overflow that already occurred during backpropagation.

Memory-saving checkpointing
---------------------------

**Activation checkpointing** recomputes intermediate states during BPTT to
save memory. It does not shorten the time window over which gradients are
calculated. It is enabled by default:

.. list-table::
   :header-rows: 1

   * - Field under ``training``
     - Default
     - Meaning
   * - ``gradient_checkpointing``
     - ``true``
     - Recompute chunks during the backward pass.
   * - ``gradient_checkpoint_chunk_size``
     - ``min(25, sequence length)``
     - Chunk length in simulation steps, not always milliseconds.
   * - ``pack_spike_checkpoints``
     - ``"auto"``
     - Bit-pack the binary spike-history state at segment boundaries into
       on-device int64 words (63 spikes per word). ``"auto"`` packs for GLIF3
       cells, whose first recurrent state is the binary spike buffer, and keeps
       other cell classes unpacked. ``false`` restores unpacked checkpoints.
       Nonzero values are treated as spikes, so only binary states are lossless.
   * - ``regenerate_initial_state_each_epoch``
     - ``true``
     - Regenerate the configured initial state at epoch boundaries.
   * - ``learning_rule``
     - ``"bptt"``
     - Learning method; the separate learning-rule example describes alternatives.

Set ``gradient_checkpointing=false`` to disable recomputation. The chunk size
must be between one and the sequence length. Smaller chunks change memory and
recomputation costs, not the intended temporal derivative. Spike packing leaves
continuous neuron states unpacked; outputs and gradients are identical to the
unpacked route. On the V1 reference workload (RTX 3090) it cost about 0.5-0.6 percent update
time at batch 1, 16 and 32 while lowering the stored boundary payload by roughly
7 MB per sample; it gives no saving at batch 1. The FP32 temporal-adjoint route
(``temporal_gradient_precision="float32"``) keeps unpacked checkpoints unless
``pack_spike_checkpoints`` is set explicitly.
The omitted chunk size follows the effective training-parameter sequence length,
including a shorter per-condition length; an explicit size is not silently
clamped. Existing legacy examples pin checkpointing off to preserve their
integer-refractory-state training route.

This is different from **saved checkpoints**, which store weights and, when
explicitly included, optimizer/state variables for later restoration.

Saved checkpoints are written in canonical SONATA edge order regardless of the
selected weight layout. With ``use_csr_master_weights`` (the automatic default on
admitted GPU routes, see :doc:`dpointnet_performance`) the live FP32 masters,
their endpoint indices and paired Adam/SGD slots are stored in CSR order, but each
is serialized as a host-converted canonical copy under the ordinary variable key;
compute-shadow entries that canonical-order models track are written as the cast
master values. Consequently ``tf.train.Checkpoint(model=..., optimizer=...)``
checkpoints restore in both directions between models built with and without
CSR masters, including checkpoints written before this option existed, and
``assert_consumed()`` holds when the remaining configuration matches. Saving never
reorders live variables. ``RNN.train`` pairs optimizer slots with their masters
after building the optimizer; custom loops that build their own optimizer call
``rnn.cell.bind_optimizer_checkpoint_slots(optimizer)`` before saving. Restoring
endpoint indices that differ from the model's connectivity raises. Keras weight
files (``save_weights``/``load_weights``) store the canonical rows as well.
Exported SONATA weights (``ModelWeights``) are always canonical.

Voltage regularization without full voltage output
--------------------------------------------------

By default, GLIF returns spikes and a compact voltage-penalty statistic rather
than every neuron's voltage at every timestep. Tracking that statistic does
**not** add a loss, clamp voltage or change the neuron dynamics.

For a configured ``VoltageRegularization`` loss, the high-level training
adapter uses an eligible compact statistic automatically. To request it
explicitly, merge this **fragment** into a complete config:

.. code-block:: json

   {
     "rnn_cell_params": {
       "track_voltage_penalty": true,
       "voltage_penalty_mode": "range",
       "return_voltage_sequences": false
     }
   }

In the loss entry under
``training.parameters[i].loss_functions``, set ``penalty_mode="range"`` and
``online=true``. Cell and loss modes must match. Compact mode supports
``range`` and ``threshold``, but not a spatial core mask.

If a loss, learning rule or analysis needs neuron-resolved voltages, set
``return_voltage_sequences=true`` instead. Full voltage tensors have shape
``[batch, time, neurons]`` and can be expensive for large models.

Monitor, save and evaluate
--------------------------

``training.callbacks`` controls loss tables, console messages and trained
weight exports. ``epoch_store_weights="latest"`` stores the latest epoch;
``"best"`` selects the lowest reported validation loss.
That score includes configured regularizers. Verify that the evaluation
conditions represent your intended held-out objective before interpreting
the saved weights as the best biological fit.

To add metrics, plots, experiment logging or early stopping, subclass
DPointNet's ``Callbacks`` rather than a Keras callback. The hook signatures,
registration and configuration are shown in :doc:`dpointnet_custom_training`.

For simulation with learned weights, copy the original inference config and
replace each ``networks.edges[i].edges_file`` with the corresponding exported
HDF5 file. In the small example these are ``glifs_glifs_edges.h5`` and
``virts_glifs_edges.h5`` under the callback's trained-weight directory.
Keep the original node files, edge-type CSVs and component directories:
the callback exports edge weights, not a self-contained replacement of all
network assets. Retain compatible dynamics and reset settings, then run the
new inference config in the original example working directory.

A SONATA weight export is not a complete optimizer checkpoint and cannot by
itself reproduce a resumed training trajectory.

SONATA edge datasets use lossless gzip compression by default, matching the
network builder. Set ``training.callbacks.sonata_compression`` to ``"gzip"``,
``"lzf"``, an integer gzip level from 1 to 9, or ``"none"``/JSON ``null`` for
uncompressed output. For example:

.. code-block:: json

   {
     "training": {
       "callbacks": {
         "class": "Callbacks",
         "sonata_compression": "none"
       }
     }
   }

When configuring callbacks explicitly, include ``"class": "Callbacks"`` (or
the name of your registered custom callback class).

For direct Python exports, pass the same selection as
``ModelWeights.to_sonata(output_dir, compression=...)``; Python ``None`` disables
compression. This applies to every edge dataset and preserves physical weights,
canonical row order, population attributes and overwrite behavior.
Compression changes storage only, not training or checkpoint semantics.

For a custom Python workflow, ``tf.train.Checkpoint`` can include the model,
optimizer and experiment state. Reconstruct the same model and loss
configuration before restoration, check that all requested objects were
restored, and refresh compute-weight copies when using custom update code.
The JSON loader does not automatically create an experiment-specific
train/validation split or restore a complete training run.

``training.callbacks.memory_report`` accepts ``epoch`` (default), ``step`` or
``off``. Reports distinguish TensorFlow allocations, driver process usage and
whole-device memory. Driver usage includes reservations; it is not a
TensorFlow live-allocation measurement. See :doc:`dpointnet_performance` when
diagnosing resource use.

Optional low-activity regularizers
----------------------------------

Neither of the following losses is enabled automatically. They alter the
objective and can bias biologically appropriate silent neurons; assess the
original fit and rate distribution as well as the added penalty.

``LowRateFloor`` penalizes the squared deficit below ``floor_hz`` (default
0.1 Hz), with ``cost`` default 1.0. Rates are pooled across the full batch and
time window. It does not replace a target-rate loss. Optional ``neuron_ids``
refer to model output columns, not arbitrary SONATA IDs. Other population/core
selection uses the visual-cortex helpers and requires the matching data.

``VoltageRateFloor`` gives subthreshold voltage credit to low-rate neurons,
using a detached moving average of firing rates. It selects all model neurons;
subset arguments are unsupported. Defaults are ``cost=1.0``, normalized
``target=0.9`` (threshold is one), ``floor_hz=0.1`` and ``ema_decay=0.95``.
These are constructor defaults, not a generally qualified rescue recipe.

For either loss, add an entry to each participating condition's
``loss_functions``. This **fragment** is one such entry:

.. code-block:: json

   {
     "voltage_floor": {
       "module": "VoltageRateFloor",
       "cost": 1.0,
       "target": 0.9,
       "floor_hz": 0.1,
       "ema_decay": 0.95
     }
   }

The voltage-rate gate starts inactive. The first accepted update initializes
it from measured rates; validation, replay and rejected updates do not update
its history. Programmatic callers must construct this loss before building
the model; JSON configuration registers its state channels automatically.
Restore the same loss configuration and history to resume it. Fused CUDA use
requires operators supporting pre-reset voltage output; older operators raise
a rebuild error. This loss requires BPTT.

Configured regularizers enter training and validation totals. If checkpoint
selection should exclude them, define that experiment policy explicitly.
They do not establish convergence for either NEST or legacy dynamics.
