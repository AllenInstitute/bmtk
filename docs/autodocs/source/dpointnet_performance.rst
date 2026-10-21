DPointNet performance and GPU setup
===================================

Start with a working example in :doc:`dpointnet_guide`. Performance depends on
network size, activity, batch, precision, inputs and outputs; there is no
universal speedup or memory requirement.

For execution across multiple GPUs, see :doc:`dpointnet_multi_gpu`.
That section covers sample splitting, batch semantics, launch and ownership;
the setup and acceleration guidance below applies to each GPU.
MPI is the recommended opt-in starting point for the qualified multi-GPU
workload; the manual also documents single-process MirroredStrategy as an
opt-in alternative. This is a workload-specific recommendation, not a
runtime-default change or universal transport-performance ranking.

Use automatic acceleration
--------------------------

For ordinary use, leave ``rnn_cell_params.acceleration_profile`` omitted
or set it to ``"auto"``. The resolver selects compatible paths based on the
GPU, loaded operators, precision, batch and network structure.
It does not change your dynamics or lower precision to fit memory.

After building a model, inspect its choices:

.. code-block:: python

   rnn.build()
   print(rnn.precision_report)
   print(rnn.acceleration_report)

These reports explain requested/resolved settings; they do not prove that
every topology-specific kernel executed. Without compatible custom operators,
automatic selection retains supported TensorFlow paths. Explicit unsupported
kernel requests raise rather than silently switching implementations.
Explicit ``acceleration_profile=null`` disables profile expansion.

Automatic state selection checks optional event-gradient, history, voltage-penalty
and type-indexed entry points individually, retaining compatible fallbacks when
a loaded library lacks them. Explicit fused NEST state requests on Pascal also
require the rebuilt occupancy-aware launch capability; disabling automatic
selection does not bypass this safety check.

Do not copy a long list of flags from a different GPU or benchmark.
For Pascal hardware, read :doc:`dpointnet_pascal` before adopting a model.
A visible NVIDIA GPU alone does not establish compatibility with every
TensorFlow or DPointNet execution route.

Projection and loss improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimized implementation reduces redundant work without changing canonical
weights, the optimizer, Poisson counts, configured dynamics or temporal-gradient
precision:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Improvement
     - What it avoids
   * - Shared immutable basis rows
     - Repeated backward projection for byte-identical effective coefficient
       rows. Edge weights and trainable physiological parameters are not shared.
   * - Warp-tiled event gradients
     - Redundant sparse weight-gradient work while retaining dense credit for
       silent sources and canonical per-edge gradients.
   * - Joint recurrent/input projection
     - Separate output buffers and launches for compatible spike sources.
       Routing metadata is cached; live weights are still read for every rollout.
   * - Compact input replay
     - Uploading FP32 inputs only to cast them to the model input dtype.
       Input producers retain count-valued Poisson events.
   * - Host canonical metadata
     - Keeping redundant canonical index/type tables on the GPU alongside native
       routing. Canonical resources and checkpoint identity are retained.
   * - Compiled firing-rate counts
     - Materializing full-sequence FP32 cast/broadcast intermediates for a rate
       objective. Only count reduction and its derivative use XLA.
   * - Compact grouped EMD indexing
     - Edge-sized host round trips during grouped weight regularization.
       Group membership, stable tie credit and the signed-int32 checkpoint schema
       are preserved.

For the standard high-level GLIF cell on one GPU, automatic joint projection
requires a compute capability targeted by the fat binaries (SM61, 70, 75, 80, 86,
89 or 90) whose SASS the loaded library carries, NEST dynamics,
compute-precision temporal gradients, FP16 or FP32 compute with FP32 masters,
four bases, a batch of 1 to 32, nonempty recurrence and one to seven supported
spike-input populations. It also checks the loaded native ABI. Batches above 32
keep the previous routes without error. Without adaptive routing, batch 1 selects
256-target tiles and larger batches 2048. A conservative 512-MiB routing-table
budget can retain 2048 or the ordinary projection instead.
The bound uses the network's maximum recurrent delay when ``max_delay`` is
``None`` or nonpositive, and accounts for each external input population's own
maximum delay independently of the recurrent cap.
These are workload-based choices, not online autotuning or a universal fastest
kernel. Other architectures/shapes retain existing compatible automatic routes.
Custom/STP cells, declared data parallelism and FP32 temporal replay do not
automatically opt into this new projection.

Joint projection accumulates sources together in FP32 and rounds once to the
compute dtype. Ordinary projection rounds each source separately before adding.
This intentional floating-point reassociation can change spikes and later
training trajectories, including in full FP32; it is not bitwise trajectory or
convergence equivalence. To retain separate projection, set:

.. code-block:: json

   {
     "rnn_cell_params": {
       "acceleration_profile": "auto",
       "use_combined_current_sweep": false
     }
   }

Inside an admitted joint sweep, automatic selection also chooses how each
source's edges are traversed, when the loaded native library has the
corresponding capability. These are scheduling choices over the same FP32
tile accumulation and single rounding; per-source gradients, live weights,
delays and Poisson counts are unchanged, although the order of atomic
additions can differ.

* ``use_cooperative_row_projection``: known target-unique sources with fewer
  than 512 edges per delay-expanded row (the recurrent history) are packed
  eight rows at a time into register-only warp work.
* ``use_fanout_row_projection``: sources with at least 512 edges per row (the
  LGN) are compacted to the rows that have edges in the block's target tile and
  walked as one concatenated edge stream, with equal-target runs reduced
  inside each warp before the shared addition.
* ``use_target_gather_projection``: sources with at most 16 incoming edges per
  target (the background input) are gathered target-major by one thread per
  target, without atomics or row lookups.

Explicit ``false`` keeps the original traversal for that source class; explicit
``true`` on an older native library raises. The automatic policy also prepares
exact per-step CPU Poisson counts once per rollout (``use_prepared_cpu_poisson``)
for the direct NEST compute-temporal loop on a single replica, when the declared
batch, rollout length and internal Poisson widths fit a 64 MiB count table.
The counts are the same draws as the per-step sampler; outside that route the
layer keeps the per-step sampler. Explicit ``true`` raises where the route is
unavailable and keeps the 64 MiB runtime budget.

CSR-ordered masters, uint8 types and adaptive routing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside an automatically admitted joint sweep, three further options default to
automatic selection. Each falls back silently to the previous path, with the unmet
prerequisites recorded in ``acceleration_report["reasons"]``; explicit ``true``
raises when unsupported and explicit ``false`` restores the previous path.

* ``use_csr_master_weights``: the recurrent and every named-input FP32 master
  variable is stored in stable CSR (source-major) edge order, and native forward
  and backward kernels read it directly as FP32. The separate FP16 and CSR compute
  shadows are removed, so no refresh or reordering happens after updates. Routing
  metadata, endpoint indices and type IDs are prepared on the host in the same
  order; one read-only ``uint32`` CSR-to-canonical map (``uint64`` above
  :math:`2^{32}` edges) is kept on the host and device canonical edge IDs are
  omitted. Requires NEST dynamics, compute temporal gradients, per-edge trainable
  recurrent weights, a single replica, a BPTT learning rule (set before the model
  is built) and the rebuilt native library.
* ``use_uint8_synapse_types``: per-edge synapse types become a byte tail of the
  same metadata resource when every type ID is in 0..255. IDs outside that range
  keep wide types; explicit ``true`` then raises instead of narrowing.
* ``use_adaptive_routing``: the joint forward sweep walks compressed per-row
  segments of nonempty target tiles (64-bit tile masks up to 64 tiles), compacts
  active rows on the device and chooses a wide or quarter-width narrow tile at
  every launch from the measured kernel occupancy and multiprocessor count.
  Sources with repeated targets use block rounds with equal-target aggregation,
  target-unique short rows use warp batches and low in-degree sources keep the
  target-major gather. With adaptive routing the automatic wide tile is 2048 at
  every batch; at batch 1 the narrow 512-target shape is normally selected. Per-edge
  arithmetic and the single FP32 tile rounding are unchanged; atomic order is not
  bitwise reproducible. Backward kernels keep their existing routing.

Implementation details are in :doc:`dpointnet_csr_masters` and
:doc:`dpointnet_adaptive_routing`. Inspect ``acceleration_report["csr_master_weights"]``,
``["csr_uint8_types"]``, ``["adaptive_routing"]``, ``["adaptive_routing_plan"]``,
``["routing_geometry"]`` and ``["launch_geometry"]``; after a warm execution,
``rnn.cell.adaptive_routing_report()`` returns the native routing-table census per
source (built tables, tile shapes and bytes). These describe selection and
allocation, not proof that every kernel executed.

With CSR masters, ``cell.recurrent_weight_values`` and
``cell.inputs[name]["input_weight_values"]`` hold live CSR-ordered rows, and so do
their gradients, constraints and optimizer slots. Use
``cell.canonical_weight_values(name)`` and ``cell.assign_canonical_weights(values,
name)`` for canonical access, and ``cell.weight_loss_indices`` /
``cell.weight_loss_reference`` to compose canonical per-edge loss metadata into
live order once before tracing. Built-in exports, ``ModelWeights``, Keras weight
files and ``tf.train.Checkpoint`` files stay canonical and interchangeable with
canonical-order models (see the saved-checkpoint notes in
:doc:`dpointnet_training`). ``eprop`` and local rules use canonical surfaces; a
learning rule configured before building keeps automatic CSR masters off.

Joint-sweep launch geometry is checked per device from its compute capability.
Tiles up to 2048 targets launch 512-thread blocks with static shared memory (at
most 43,184 bytes per block for every combined-sweep kernel on all seven targets,
64 registers and no spills), which is valid on every target. Wider tiles use
dynamic shared memory and 1,024-thread blocks for 32-bit routing: SM61 (48 KiB per
block) supports none of them, SM75 (64 KiB) only 3072, and SM70, SM80, SM86, SM89
and SM90 all three. Automatic selection falls back to the widest valid tile; an
explicit tile that a device cannot launch raises before construction.

Advanced explicit joint selection uses ``use_combined_current_sweep=true`` and
``combined_current_target_tile_size`` in ``rnn_cell_params``. Allowed tile sizes
are 256, 512, 1024, 2048 and the wide 3072, 4096 and 5120. Wide tiles need
device-supported opt-in shared memory and launch 1,024-thread blocks for 32-bit
routing; the automatic choice remains 256 (batch 1 without adaptive routing) or
2048. On the V1 reference
workload a 5120 tile is faster for the LGN alone but slower for the full
update, because the background source costs more per block there.
Explicit choices retain validation and are not
evidence that an unqualified hardware/feature combination is supported.
Small tiles use more routing memory and are not a general large-batch tuning
recommendation.

``TargetFiringRate.compiled_spike_counts`` defaults to ``"auto"``. It selects the
compiled FP32 count reduction after the model is built for a declared standard
single-replica SM86/native model on TensorFlow 2.21. Other models retain the
ordinary reduction. Explicit ``true``/``false`` override this choice; put the
option alongside ``module`` and ``firing_rate`` in the loss configuration.
Explicit compilation failures are reported, not silently replaced with another
objective. This feature does not compile the entire RNN or accelerate every loss.
Custom callers should feed compact spikes directly rather than first casting the
entire sequence to FP32.

The same admitted high-level native model prepares its live type-identity
predicate on the compute device once per rollout, outside custom-gradient
carrier arguments. Timestep transitions reuse that predicate to select compact
or full-coefficient execution, falling back when within-type equality is lost.
Automatic selection does not enable the static-dispatch promise.

Precision and memory
--------------------

The switches in :doc:`dpointnet_training_standards` control compute precision
and state/gradient accuracy. Full voltage sequences are optional and often
expensive; use :doc:`compact voltage regularization <dpointnet_training>`
when your objective permits it.

Batch size is an experimental choice, not an automatic speed setting.
Some implementations have specialized batch/basis requirements; other
shapes retain general paths. Measure initialization, representative activity
and optimizer updates with memory headroom, not just an idle model.
Changing condition layout can change the number and meaning of updates.

``use_device_poisson`` remains false by default. Its device sampler is still
Poisson, but changes seeded realizations. Treat enabling it as an input-policy
change and check replay, rather than assuming it is an identical speed switch.

FP32 derivatives across time
----------------------------

To check weak or long-lag learning signals under mixed precision, merge this
**JSON fragment** into a complete configuration:

.. code-block:: json

   {
     "run": {
       "precision": {
         "mixed_precision": true,
         "fp32_voltage_and_asc": true,
         "fp32_temporal_gradients": true
       }
     }
   }

This uses FP32 derivatives across time while retaining the chosen forward
precision. It is not equivalent to a full FP32 simulation and does not widen
all synaptic forward state.

The supported temporal runner records projected synaptic currents so backward
replay uses the original forward trajectory. Omitted/null
``rnn_cell_params.current_replay_mode`` resolves to ``"record"`` for this
route. Explicit ``"recompute"`` saves that recording but repeats current
projection and is approximate when GPU reductions are nondeterministic.
Explicit replay modes require FP32 temporal gradients.

Recording requires approximately
``batch * neurons * steps * basis_count * bytes_per_current`` of host memory,
in addition to the model. For FP16 currents, ``bytes_per_current=2``.
Chunked replay reduces per-transfer size, not this total recording.
Measure host RAM, GPU memory and transfer costs on your model.

FP32 temporal gradients require selective state and a supported
``TemporalAdjointRunner`` or ``ExplicitStateRNN``. A direct cell tape or a
custom loop does not become FP32-temporal just by setting the option.
Do not force FP16 packed backward kernels, nest another gradient-replay
runner around it, or assume independent GPU forwards become bitwise equal.
Masked, time-major, backwards, stateful and explicitly unrolled execution
are unsupported by this route. Custom-runner details are in
:doc:`dpointnet_development`.

Build optional CUDA operators
-----------------------------

Skip this section for the CPU first-run example. For custom GPU acceleration,
build in the same environment used to execute BMTK. You need GPU-enabled
TensorFlow, a CUDA toolkit containing ``nvcc`` and a compatible C++17 compiler.
The toolkit/compiler must match TensorFlow's build requirements; installing
TensorFlow's runtime CUDA packages alone does not supply the full build setup.

From the intended checkout:

.. code-block:: bash

   python -c 'import bmtk; print(bmtk.__file__)'
   nvcc --list-gpu-code
   python -m bmtk.simulator.dpointnet.custom_ops.build

The build produces both ``_csr_spike_ops.so`` (synaptic-current operations)
and ``_glif_state_ops.so`` (neuron-state operations). Inspect their
``.archs`` manifests to check which GPU architectures were included.
Use the current checkout, not an unrelated installed BMTK.

For a shared installation spanning Pascal through Hopper, explicitly build
a **fat binary**: one operator library containing code for multiple GPU
architectures. This is a deployment option, not a first-run requirement:

.. code-block:: bash

   DPOINTNET_CUDA_ARCHS="61 70 75 80 86 89 90" \
     python -m bmtk.simulator.dpointnet.custom_ops.build

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Compute capability
     - Example hardware
   * - 6.1
     - GTX1080Ti / Titan Xp
   * - 7.0
     - V100
   * - 7.5
     - Quadro RTX8000
   * - 8.0
     - A100
   * - 8.6
     - RTX3090
   * - 8.9
     - L40S
   * - 9.0
     - H200

Keep targets ascending and space-separated. The final target also gets PTX,
an intermediate GPU code format (PTX90 for this list).
Pascal 6.1 requires an explicit target; it is not in the implicit build list.
If the compiler cannot cover your intended devices, use a compatible toolchain
or separate declared builds; do not silently omit a required device.

These libraries are reusable across included GPUs only with compatible
TensorFlow/CUDA binary interfaces. Rebuild after CUDA source, signature or
binary-interface changes. Do not replace a library in use by another process.
On clusters, follow the site's rules for compilation and compute allocations.

.. warning::

   A successful multi-architecture build is not a test of every device.
   Check the actual model, inputs, precision, updates, restoration and memory
   on each new GPU family. Pascal automatic routing requires the repaired,
   capability-marked state operators described in :doc:`dpointnet_pascal`;
   physical multi-GPU execution is not established by the build.

Measure an improvement
----------------------

Compare the same hardware, inputs, activity, precision, losses, update protocol
and output mode. Exclude tracing/compilation and warm up before timing.
For repeatable measurements, use at least three warmups and twenty synchronized
samples, and report variation as well as a median.

Separate input generation, the model update, validation and result writing.
TensorFlow allocation differs from driver memory reservations; neither is a
standalone fragmentation measure. See the training callback's memory reports.
Default launchers use TensorFlow's BFC memory allocator; importing BMTK does
not override an application's allocator environment.
