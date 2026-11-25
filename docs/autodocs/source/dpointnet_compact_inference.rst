Memory-efficient frozen inference
=================================

``RNN.from_config`` automatically selects compact recurrent connectivity for a
supported inference-only configuration. A nonempty ``training`` section retains
the standard training representation, including a configuration that subsequently
runs inference. No training options, input events, timestep, reset, precision or
edge selection are changed to obtain a fit.

Merge this optional fragment into a complete configuration:

.. code-block:: json

   {
     "run": {
       "inference_connectivity": "auto"
     }
   }

``auto`` is the default. ``standard`` preserves the existing representation and
is the regression/compatibility override. ``compact`` requires the frozen
backend and raises if it is unsupported; it never silently substitutes another
backend. Selection and its reason are logged during construction and available
as ``rnn.connectivity_report``. Missing native binaries do not trigger a build
or installation.

Eligibility and execution
-------------------------

The initial integration supports the built-in ``GLIF3Cell``, NEST dynamics,
FP32 or mixed-FP16 compute with the existing state-precision choice, and a
single distribution replica. Batch size and synaptic basis width are not
hardcoded; target/synapse-type packing must fit uint32. Duplicate basis rows
are deduplicated after conversion to the requested compute dtype, without
changing the original basis used by external inputs.

Custom cell subclasses, per-type recurrent scaling, temporal-adjoint execution,
legacy delay layouts, multiple replicas, neuron-partitioned (``mpirun``) execution,
enclosing strategy scopes and explicit
recurrent-accelerator selections other than ``false`` retain the standard path
in ``auto``. Explicit
``acceleration_profile=null`` or ``use_fused_cuda=false`` also retains the
reference path. Unsupported numeric precision is not narrowed automatically.

For the Python API, execution intent is not inferred merely from the temporary
absence of a training engine: a model can be built before training is attached.
API models therefore retain standard connectivity in ``auto``. Explicitly
construct ``RNN(..., inference_connectivity="compact")`` for frozen execution.
Direct ``GLIF3Cell`` construction retains its existing behavior.

Compact models reject attaching a training engine, executing the RNN with
``training=True``, or differentiating through the recurrent projection.
Rebuild a separate standard model for training. Evaluating an existing trained
model does not rewrite its connectivity or discard training state.

Device and host storage
-----------------------

The device stores:

* int64 source/delay CSR row offsets;
* one uint32 packed target/synapse-type value per recurrent edge;
* one compute-dtype recurrent weight per edge;
* a small deduplicated synaptic basis table.

There is no device canonical endpoint table, edge permutation, FP32 recurrent
master, sign mask or backward-pair metadata in this backend. With FP32/FP16
compute, recurrent layout storage is respectively 8/6 bytes per edge plus
8 bytes per row offset and the basis table. This is not total GPU memory:
neuron state, delay history, external inputs, current accumulators, upload
temporaries and allocator reservations also need headroom.

Canonical FP32 weights and the canonical-to-compute permutation remain on the
CPU for export and restoration. The cell constructs compact device connectivity
directly instead of first allocating the standard GPU representation.
The existing SONATA loader still retains host-side source arrays; this feature
is not the separate cortex project's two-pass streaming/cache loader and does
not promise its host-memory reduction.

``get_model_weights`` continues to return canonical normalized weights and
physical SONATA export uses the original per-edge scale factors. The canonical
master is frozen/nontrainable, not silently downcast for export. The GPU
compute copy uses the precision already requested by the configuration;
it is not an independent FP16-weights/FP32-dynamics precision option.

Canonical assignment and checkpoints
------------------------------------

``run_inference`` refreshes compact compute weights from the canonical CPU
master before execution, including after checkpoint restoration or explicit
assignment. This transfer is outside the timestep loop. A custom runner using
the cell or ``run_extractor`` directly must call
``cell.refresh_recurrent_weight_shadow()`` after changing/restoring masters.
Do not refresh on every temporal chunk of an unchanged continuous trajectory.

TensorFlow checkpoints of the compact backend can be restored into the same
backend/layout. Standard and compact TensorFlow checkpoint object graphs are
different: retain ``inference_connectivity="standard"`` when directly restoring
a standard training checkpoint. To construct a new compact model from trained
weights, export canonical SONATA weights and load that network through the
ordinary configuration. Do not hide unmatched checkpoint state with a partial
restore. No change is made to the existing training checkpoint format.

Build and validation
---------------------

The explicit native build in :doc:`dpointnet_performance` also builds
``_compact_inference_ops.so`` and its architecture manifest. Match the
TensorFlow/compiler/CUDA ABI, and rebuild after source or ABI changes.
The compact library supplies CPU and CUDA kernels; GPU selection checks the
visible device against the compiled architecture manifest. A CPU-only build
is not GPU qualification.

Current projection accumulates in FP32 before rounding once to compute dtype.
GPU atomic reductions can differ in floating-point association from the
standard projector; compact storage is not a claim of bitwise equality for
arbitrary trajectories. Test representative states/events and complete chunk
carry against a standard reference before adopting a scientific workload.
Memory reduction is the goal, not a universal speedup.
