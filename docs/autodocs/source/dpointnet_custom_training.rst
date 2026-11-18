DPointNet custom losses and callbacks
=====================================

Custom objectives and training callbacks are ordinary parts of an experiment.
You do not need to edit BMTK to add them. This page shows the complete path:
define a Python class, register its name, select it in JSON, and import the
registration code before loading the model.

Start in the writable copy of ``examples/dpointnet_all2all`` used in
:doc:`dpointnet_guide`. The example below retains that network and its
``rand_inputs`` generator. See :doc:`dpointnet_training` for the general
training configuration.

Define and register the extensions
----------------------------------

Create ``custom_training.py`` in the example working directory with the
following contents:

.. code-block:: python

   import json
   import math
   from pathlib import Path

   import tensorflow as tf
   from bmtk.simulator.dpointnet import add_loss_module
   from bmtk.simulator.dpointnet.callbacks import Callbacks, add_callbacks


   class MeanRateMSE:
       def __init__(self, rnn, target_hz, cost=1.0, **kwargs):
           self.dt = float(rnn.dt)
           self.target_hz = float(target_hz)
           self.cost = float(cost)
           if not math.isfinite(self.dt) or self.dt <= 0:
               raise ValueError("dt must be finite and positive")
           for name in ("target_hz", "cost"):
               value = getattr(self, name)
               if not math.isfinite(value) or value < 0:
                   raise ValueError(f"{name} must be finite and nonnegative")

       def __call__(self, spikes, **kwargs):
           spikes = tf.cast(spikes, tf.float32)
           rates_hz = tf.reduce_mean(spikes, axis=(0, 1)) * (1000.0 / self.dt)
           return self.cost * tf.reduce_mean(tf.square(rates_hz - self.target_hz))


   add_loss_module(MeanRateMSE, module_name="MeanRateMSE")


   class StepTraceCallbacks(Callbacks):
       def __init__(self, rnn, run_label="trial", stop_below=None, **kwargs):
           super().__init__(rnn=rnn, **kwargs)
           self.run_label = run_label
           self.stop_below = None if stop_below is None else float(stop_below)
           if self.stop_below is not None and not math.isfinite(self.stop_below):
               raise ValueError("stop_below must be finite or None")
           self.trace_path = Path(self.callbacks_dir) / "step_events.jsonl"

       def on_train_begin(self):
           super().on_train_begin()
           self.trace_path.parent.mkdir(parents=True, exist_ok=True)
           self.trace_path.write_text("", encoding="utf-8")

       def on_step_end(self, loss_vals):
           super().on_step_end(loss_vals)
           record = {
               "run_label": self.run_label,
               "epoch": self.epoch_num,
               "step": self.epoch_step_num,
               "total_loss": float(loss_vals["__total_loss"]),
           }
           with self.trace_path.open("a", encoding="utf-8") as stream:
               stream.write(json.dumps(record, allow_nan=False) + "\n")

       def on_epoch_end(self, validation_losses):
           stop = super().on_epoch_end(validation_losses)
           score = float(validation_losses["__total_loss"])
           if not math.isfinite(score):
               raise ValueError("Non-finite validation total")
           return bool(stop) or (
               self.stop_below is not None and score < self.stop_below
           )


   add_callbacks(StepTraceCallbacks, name="StepTraceCallbacks")

The custom loss fits each neuron's mean firing rate, pooled across batch and
time, to one scalar target in Hz. It is not identical to the built-in
``TargetFiringRate``, which compares rates separately for each trial.
The custom callback adds a JSON-lines trace and optional epoch-level stopping
while preserving the standard loss tables and weight exports.

Loss interface
--------------

The JSON loader constructs a loss as ``LossClass(rnn=network, **loss_entry)``.
The constructor therefore needs ``rnn``, the configurable arguments it uses,
and ``**kwargs`` for loader fields such as ``module`` and ``enabled``.
Explicit registration above supplies the lookup name; no Keras loss subclass
or ``module()`` method is required when ``module_name`` is given.

During training and validation, DPointNet calls the object with keyword
arguments including ``spikes``, ``voltages``, ``model_state`` and ``y``.
Use ``__call__(self, spikes, **kwargs)`` for a spike-only loss. This is not
Keras's ``loss(y_true, y_pred)`` interface; ``y`` is input-generator metadata,
not an automatically loaded target array.

* ``spikes`` has shape ``[batch, time, neurons]``. Output columns follow the
  model's neuron ordering, not arbitrary SONATA node IDs.
* Return one differentiable scalar tensor. The training engine adds the
  configured loss terms; coefficients such as ``cost`` belong to your loss.
* Keep computation in TensorFlow. Converting spikes to NumPy or Python values
  inside the loss breaks the gradient path. The example casts to FP32 without
  detaching gradients, including when model computation uses mixed precision.
* Load fixed target files and construct masks in ``__init__``, then keep the
  loss calculation free of file I/O and Python-side history updates.
  Graph tracing, validation and checkpoint recomputation can invoke the loss
  outside the sequence of accepted optimizer updates.
* For a loss needing neuron-resolved voltages, explicitly set
  ``rnn_cell_params.return_voltage_sequences=true``. The default compact
  voltage statistic is not a full ``[batch, time, neurons]`` voltage tensor.

Use a unique registration name. Reusing a built-in name replaces its
implementation in that Python process; that should be an intentional choice,
not a way to select a different objective accidentally.

.. _dpointnet-streaming-custom-losses:

Streaming-capable custom losses
-------------------------------

``MeanRateMSE`` above receives the dense ``[batch, time, neurons]`` spike
tensor, so a configuration using it runs on the dense route and the training
log says ``Training route: dense spike sequences; ... MeanRateMSE: undeclared``.
The fast routes (checkpointed streaming and the direct recorded adjoint, see
:doc:`dpointnet_training`) never materialize that tensor; they hand losses a
``SpikeSummary`` of FP32 spike counts over declared windows plus declared trace
plans. A project loss joins those routes by declaring the
``StreamingSpikeLoss`` contract; nothing in BMTK needs to change:

.. code-block:: python

   import numpy as np
   import tensorflow as tf
   from bmtk.simulator.dpointnet import StreamingSpikeLoss, register_loss_module
   from bmtk.simulator.dpointnet.spike_summary import SpikeSummary
   from bmtk.simulator.dpointnet.spike_statistics import ProjectedSpikeTraces


   @register_loss_module(module_name="ReadoutEnergy")
   class ReadoutEnergy(StreamingSpikeLoss):
       """Mean squared projected activity over a window, plus a count penalty."""

       streaming_plan_kinds = ("counts", "projected traces")

       def __init__(self, rnn, projection_file, window, cost=1.0, pre_delay=50, **kwargs):
           self.projection = tf.constant(np.load(projection_file), tf.float32)  # [neurons, features]
           self.window = tuple(window)
           self.cost = float(cost)
           self._pre_delay, self._post_delay = int(pre_delay), 0  # default count_windows() reads these

       def streaming_unmet(self, context):
           if context.sequence_length is not None and self.window[1] > context.sequence_length:
               return (f"ReadoutEnergy window {self.window} exceeds the sequence length.",)
           return ()

       def prepare_spike_statistics(self, batch_size, dtype, sequence_length, batch_start=0):
           return ProjectedSpikeTraces(self.projection, self.window, (batch_start, batch_start + batch_size))

       def __call__(self, spikes, **kwargs):
           if isinstance(spikes, SpikeSummary):
               counts = spikes.trim(self._pre_delay, self._post_delay).temporal_sum()
               traces = spikes.trace_for_loss(self.name)          # [batch * features, window]
           else:                                                   # dense route, e.g. validation
               counts = tf.reduce_sum(tf.cast(spikes[:, self._pre_delay:], tf.float32), axis=1)
               start, stop = self.window
               traces = tf.reshape(tf.einsum(
                   "btn,nf->bft", tf.cast(spikes[:, start:stop], tf.float32), self.projection,
               ), [-1, stop - start])
           return self.cost * (tf.reduce_mean(tf.square(traces)) + 1e-3 * tf.reduce_mean(counts))

The contract:

* ``supports_streaming`` (class attribute, ``True`` from the mixin) is the
  declaration. Set it to ``False`` to keep a subclass of a streaming loss on the
  dense route explicitly. A loss without the attribute is *undeclared*.
* ``streaming_unmet(context)`` returns user-facing sentences when this instance
  cannot stream in ``context`` (``sequence_length``, per-group ``batch_sizes``,
  ``cell``, ``neuron_partition``, ``replicas``); an empty tuple admits it.
* ``count_windows(sequence_length)`` lists the ``(start, stop)`` count windows the
  loss reads with ``SpikeSummary.trim(...).temporal_sum()``; the default uses
  ``_pre_delay``/``_post_delay`` when present. The full window is always there.
* ``prepare_spike_statistics(batch_size, dtype, sequence_length, batch_start)``
  returns at most **one** trace plan for one parameter group: a
  ``SampledSpikeTraces`` (sparse per-neuron traces, as the synchronization loss
  uses) or a ``ProjectedSpikeTraces`` (``spikes @ P`` for a static FP32
  projection), covering exactly the group's batch rows; ``()`` for counts only.
  Plans are data interpreted by BMTK's runners, never callbacks, so the streaming
  kernels, memory bounds and credit paths stay in BMTK while the objective stays
  in your project. The loss reads its plan back with
  ``spikes.trace_for_loss(self.name)`` (``name`` is the configuration key unless
  the loss defines one; plans are keyed by it, hence one plan per loss — a loss
  needing several statistics is configured as several losses) and may appear once
  per parameter group.
* ``streaming_plan_kinds`` describes the plans for the route report
  (``"counts"``, ``"sampled traces"``, ``"projected traces"``).
* ``__call__`` must accept a ``SpikeSummary``; keep a dense branch if the loss is
  also used where full spike tensors are produced.

Verify a streaming loss the way BMTK verifies its own: run a small network through
the dense route (``stream_spike_statistics: false``) and through ``"auto"`` and
compare values and gradients (``tests/simulator/dpointnet/test_streaming_loss_protocol.py``
is a template). With the declaration in place, ``"auto"`` selects the fastest
route the loss qualifies for, and the route line names it: ``Training route:
checkpointed streaming + direct recorded adjoint; ...; losses: readout
[ReadoutEnergy: declared by my_project.losses, counts, projected traces]``.
Assert ``rnn.training_engine.training_route_report["mode"]`` in configurations
that expect the fast route.

To register from configuration without an import in the launcher, list the
modules under ``training.loss_modules``; they are imported (and therefore
registered) before any loss is looked up:

.. code-block:: json

   "training": {
     "loss_modules": ["my_project.losses"],
     "parameters": [
       {
         "name": "evoked",
         "inputs": ["lgn"],
         "loss_functions": {
           "readout": {
             "module": "ReadoutEnergy",
             "projection_file": "readout_projection.npy",
             "window": [50, 500],
             "cost": 1.0
           }
         }
       }
     ]
   }

Callback interface
------------------

DPointNet uses its own ``Callbacks`` interface, not
``tf.keras.callbacks.Callback``. Subclass the DPointNet class and forward
``rnn`` and standard options to ``super().__init__``. For hooks you override,
call the parent hook to retain logging, counters, timings, saved weights and
exports. Unchanged hooks are inherited automatically.

.. list-table:: Training hooks
   :header-rows: 1
   :widths: 45 55

   * - Method
     - When it runs
   * - ``on_train_begin()``
     - Once before the first epoch.
   * - ``on_epoch_start()``
     - Before an epoch; the parent increments ``epoch_num``.
   * - ``on_step_start()``
     - Before a training step; the parent increments step counters.
   * - ``on_step_end(loss_vals)``
     - After the training step, with its loss and metric dictionary.
   * - ``on_epoch_end(validation_losses)``
     - After evaluation; returning true stops the epoch loop.
   * - ``on_train_end(metrics=None, normalizers=None)``
     - At normal training completion; the parent exports saved weights.

Both loss dictionaries contain ``"__total_loss"`` and nested dictionaries
such as ``loss_vals["custom_rates"]["mean_rates"]`` for named terms.
Reserved metric keys begin with ``__``. Convert scalar tensors to Python
values for host-side logging, as the callback does, not inside the loss.

The example's ``stop_below=None`` disables early stopping. A numeric threshold
compares the reported validation total, including regularizers; it does not
create a held-out split. Choose the stopping score for your experiment.
Use a new callback output directory for a new run: this example resets its
trace at ``on_train_begin`` and does not resume a previous training trajectory.

Select the extensions in configuration
--------------------------------------

Save the following **JSON fragment** as ``custom_training_config.json``.
The launcher below merges its ``training`` fields into the complete
``config.train.all.json``. Thus the original network, inputs, initial state,
two epochs and three steps per epoch remain available.

.. code-block:: json

   {
     "training": {
       "training_approach": "single",
       "gradient_checkpointing": false,
       "learning_rate": 0.001,
       "callbacks": {
         "class": "StepTraceCallbacks",
         "callbacks_dir": "custom_training_outputs",
         "run_label": "rate_trial",
         "stop_below": null,
         "verbose": "on_epoch",
         "memory_report": "off",
         "epoch_store_weights": "latest",
         "sonata_output_dir": "trained_network",
         "losses_table_csv": "losses.csv"
       },
       "parameters": [
         {
           "name": "custom_rates",
           "inputs": ["rand_inputs"],
           "loss_functions": {
             "mean_rates": {
               "module": "MeanRateMSE",
               "target_hz": 15.0,
               "cost": 0.01
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

Here ``"mean_rates"`` is a reporting label, whereas ``"MeanRateMSE"`` is
the registered implementation name. Losses belong in each condition's
``loss_functions``; ``callbacks`` belongs under ``training`` and selects
one callback class. Combine additional hooks in that class rather than
supplying a Keras callback list.

The coefficients are illustrative. Checkpointing is disabled only for this
legacy example's refractory-state dtype limitation, explained in
:doc:`dpointnet_training`; it is not a general requirement for custom losses.

Import registration code and run
--------------------------------

Save this as ``run_custom_training.py`` beside the other two files:

.. code-block:: python

   import json
   import os
   from pathlib import Path

   os.environ.pop("TF_GPU_ALLOCATOR", None)

   import custom_training
   from bmtk.simulator import dpointnet


   def main():
       config = dpointnet.Config.from_json("config.train.all.json")
       fragment = json.loads(Path("custom_training_config.json").read_text())
       config["training"].update(fragment["training"])
       config.build_env()
       rnn = dpointnet.RNN.from_config(config)
       try:
           rnn.run()
       finally:
           rnn.cleanup()


   if __name__ == "__main__":
       main()

Run from the example directory, using the environment with your intended
BMTK installation:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="" python run_custom_training.py

Importing ``custom_training`` executes both registrations **before**
``RNN.from_config`` looks up the JSON names. JSON does not import an arbitrary
Python file automatically. Launching the unchanged generic entrypoint in
another process will not inherit these registrations.

With early stopping disabled, the example makes six optimizer updates and
then runs inference. Check:

* ``custom_training_outputs/step_events.jsonl``: six records containing the
  run label, epoch, step and total loss.
* ``custom_training_outputs/losses.csv``: individual ``mean_rates`` and
  ``voltage`` terms as well as totals.
* ``custom_training_outputs/trained_network/``: exported trained edge files;
  see :doc:`dpointnet_training` for reusing them with the original network assets.

For a missing loss-module or callback-class error, first check the spelling
and whether the registration module was imported in the same Python process
before model loading. For disconnected gradients, check that the loss returns
a tensor and does not use NumPy, ``.numpy()`` or ``stop_gradient`` on the
prediction path. Test the scalar value and gradients on a small known tensor
before applying a new loss to a large network.
