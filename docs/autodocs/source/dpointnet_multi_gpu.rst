DPointNet multi-GPU support
==============================

This page describes DPointNet's opt-in **sample-parallel training** support.
Start with :doc:`dpointnet_guide` for a working network and
:doc:`dpointnet_training` for objectives, optimizers and evaluation.
Use documentation from the same BMTK revision as your installation.

Each GPU holds a complete copy of the network and processes different input
trials. The replicas combine their gradients to make one shared weight update:

.. code-block:: text

   Global batch of input trials
              |
       split by sample
         /          \
     GPU 0          GPU 1
   full network   full network
   local trials   local trials
         \          /
          mean gradient
               |
       one shared update

This increases the number of trials processed per update and can reduce
the per-GPU memory needed for a given global batch. It does **not** divide
neurons or connections between GPUs, and it does not pool GPU memory:
the complete network and each local batch must still fit on every device.
Both execution modes below are single-host engineering previews; omission
of ``run.distribution`` preserves existing behavior.

Choose an execution mode
------------------------

**Start with MPI when opting into multi-GPU training.** One TensorFlow process
per GPU is the recommended approach for the qualified DPointNet workload.
Single-process ``MirroredStrategy`` remains an explicitly selected alternative.
This is the manual's recommended starting point, **not an automatic runtime
default**: neither mode is enabled merely because several GPUs are visible.
Omitting ``run.distribution`` preserves the previous execution and batch behavior.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Approach
     - Explicit selection
     - When to use it
   * - MPI (recommended)
     - ``strategy="mpi"`` plus the MPI GPU-selection launcher
     - Starting point for supported single-host multi-GPU training
   * - MirroredStrategy (alternative)
     - ``strategy="mirrored"`` in one TensorFlow process
     - An alternative to qualify and benchmark for your workload

In the qualified two-V10032 full-V1 tests, optimized MPI achieved
**1.881528x throughput scaling**, while optimized single-process
MirroredStrategy achieved **1.551438x**. Each result used its own matched
single-GPU control in a separate qualification run; these are not a simultaneous
head-to-head benchmark. They support the MPI recommendation for that workload,
not a claim that MPI transport is intrinsically faster than TensorFlow's
collectives or that every network and GPU will scale the same way.

Sample-parallel training
------------------------

An explicit ``run.distribution`` opts into equal, disjoint sample batches on
one host. Both modes hold the complete network on every replica; neither
partitions neurons or exchanges spikes between GPUs.

``run.batch_size`` and the sole training condition's ``batch_size`` are
**per-replica** sizes. The optional ``global_batch_size`` must equal this size
times the selected replica count. There is one optimizer update per global
batch. Device ordinals refer to TensorFlow's visible allocation, not arbitrary
physical GPU numbers. Explicit missing, duplicate or remote devices raise.

The initial scope is full FP32, BPTT, one condition, equal full batches and
zero-state initialization. Mixed precision, condition ``parallel``/``series``/
``series_accumulate``, local learning rules, warmup/cached initial states and
optimizer clipping are rejected until separately qualified. No sample
microbatch-accumulation option is implied by condition accumulation.

``TargetFiringRate`` is a sample-separable mean and is admitted. Its existing
``rate_units="Hz"`` default is unchanged. Explicit
``rate_units="spikes_per_step"`` compares each sample/neuron's time-mean spike
count with ``firing_rate`` in spikes per simulation step; ``0.005`` matches the
engineering prototype's probability-space objective at 1-ms steps. This is
not the same loss scale as an Hz-space squared error. Select units deliberately.
Other objectives must declare and independently qualify
``data_parallel_reduction="replica_mean"``; nonlinear population/batch
statistics cannot in general be computed by averaging local scalar losses.
Unqualified objectives raise instead of silently changing the objective.

The training iterator produces one global host batch using the input modules'
existing RNG/seed configuration, then splits inputs and targets into contiguous,
disjoint sample slices. Each slice is explicitly placed on its replica before
the temporal loop. Creating ``PerReplica`` values alone does not establish
device residence: a peer's full batch can otherwise be copied repeatedly
inside checkpoint/recomputation functions. This preview
intentionally uses the host/reference generator instead of the existing
per-device LGN iterator, whose historical semantics replicate a parameter
batch. Internal Poisson noise retains replica-specific replay-safe streams;
changing the replica count is not a bitwise noise-reproduction guarantee.

Build both operators for the intended pool before running. Compilation and
logical-CPU tests alone do not qualify two physical GPUs. Record the resolved
distribution, precision, acceleration report and actual operator devices.

Every replica refreshes recurrent and trainable input compute shadows after
the update. Checkpoints must explicitly include all canonical masters,
including frozen named-input weights, and optimizer/noise state; validate
restoration by mutating and checking those resources, not only by accepting
``assert_consumed()``.

This preview is not a package release, convergence claim, all-GPU qualification,
multi-node backend or replacement for network/neuron-partitioned simulation.

One process per GPU (MPI engineering preview)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``run.distribution.strategy="mpi"`` explicitly selects the **recommended**
single-host execution mode. Each process owns one complete network on its sole visible GPU.
Only sample batches are split; neurons, connections and temporal state are
not partitioned. Omission and explicit ``"mirrored"`` keep their existing
behavior. This is a separately qualified opt-in backend, not a promise of
near-linear scaling on every workload.

For two processes with local batch 16, use this **configuration fragment**:

.. code-block:: json

   {
     "run": {
       "dtype": "float32",
       "batch_size": 16,
       "distribution": {
         "strategy": "mpi",
         "devices": ["/GPU:0"],
         "global_batch_size": 32
       }
     }
   }

Launch inside a two-GPU allocation with an MPI runtime and ``mpi4py`` from
the selected environment:

.. code-block:: bash

   mpirun -n 2 python -m bmtk.utils.dpointnet_mpi run_dpointnet.py config.json

``mpirun`` is the traditional spelling in BMTK examples. ``mpiexec`` is also
valid when provided by the selected MPI installation; most installations expose
both as equivalent launchers. Use the launcher compatible with the MPI library
used by ``mpi4py`` rather than mixing Open MPI and MPICH installations.
The required extra step is ``python -m bmtk.utils.dpointnet_mpi``, which assigns
each process its GPU before TensorFlow import, regardless of launcher spelling.

The optional package extra is ``bmtk[dpointnet-mpi]``; installing the extra
does not select or qualify an MPI runtime, TensorFlow/CUDA ABI or GPU family.
Keep existing immutable environments unchanged. The launcher selects rank
ordinals **within inherited** ``CUDA_VISIBLE_DEVICES`` before TensorFlow
import. Never replace a Slurm allocation such as ``2,3`` with physical ``0,1``.
Every process then calls its assigned GPU ``/GPU:0``. The inherited list
must contain exactly one distinct GPU per process. Multi-host jobs are rejected.
The launcher's explicit ``--cpu`` option is for CPU qualification.

Runnable configuration example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source checkout includes ``examples/dpointnet_mpi/config.train.json``
and ``run_training.py``. They reuse the checked-in 300-neuron SONATA network
and cell/synapse models in ``examples/dpointnet_all2all``; no network download
or regeneration is required. Run these commands from the checkout root with
an MPI runtime, TensorFlow and the optional ``dpointnet-mpi`` extra installed:

.. code-block:: bash

   # CPU smoke: two processes, local batch 2, global batch 4.
   mpirun -n 2 python -m bmtk.utils.dpointnet_mpi --cpu \
     examples/dpointnet_mpi/run_training.py --output /tmp/dpointnet-mpi

   # Ordinary CPU control, without the new distribution configuration.
   CUDA_VISIBLE_DEVICES="" python examples/dpointnet_mpi/run_training.py \
     --serial --output /tmp/dpointnet-serial

Use separate, fresh output directories. The example selects FP32, NEST
hard-forward/soft-surrogate dynamics, 64 timesteps, checkpoint interval 16,
zero initial state, seeded **Poisson** inputs and a target-rate objective in Hz.
It performs two epochs of two Adam updates and logs
``Completed 4 optimizer updates.`` once. The copied resolved configuration
and ``run.log`` appear in the output directory. Chief callbacks write
``callbacks/losses.csv``, ``callbacks/performance.csv`` and SONATA edge files
under ``callbacks/trained_weights``. These are best-epoch weight exports,
not a complete optimizer/noise resume checkpoint.
The serial control retains batch 2, whereas MPI combines batch 4;
these commands are smoke tests, not a matched gradient or speed comparison.
Four updates do not establish convergence or biological accuracy.

For the GPU version, build both native operators for the selected
TensorFlow/CUDA environment as described in :doc:`dpointnet_performance`.
Inside an allocation exposing exactly two qualified GPUs, omit ``--cpu``:

.. code-block:: bash

   mpirun -n 2 python -m bmtk.utils.dpointnet_mpi \
     examples/dpointnet_mpi/run_training.py --output /path/to/new/results

A single-node Slurm allocation can run one MPI launcher which starts both
TensorFlow processes:

.. code-block:: bash

   #!/usr/bin/env bash
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=8
   #SBATCH --gres=gpu:2
   #SBATCH --mem=16G
   #SBATCH --time=00:05:00
   set -euo pipefail
   # Activate the qualified environment and cd to the source checkout first.
   export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=4
   export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
   mpirun -n 2 python -m bmtk.utils.dpointnet_mpi \
     examples/dpointnet_mpi/run_training.py --output /path/to/new/results

Partition/account, MPI process-manager integration and GPU/host-memory sizing
are site-specific; these resource sizes cover the small example, not V1.
Both child processes must inherit the **full two-GPU allocation list**.
A launch mode which restricts each process to one token before this launcher
runs is not supported. Do not override the scheduler's GPU assignment.

Normal BMTK configuration and output ownership
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every rank executes the same normal entry point:

.. code-block:: python

   from bmtk.simulator import dpointnet

   config = dpointnet.Config.from_json("config.json")
   config.build_env()
   network = dpointnet.RNN.from_config(config)
   network.run()

Under the MPI launcher, DPointNet's I/O adapter obtains rank, size and
barriers from BMTK's shared communicator. Rank zero alone creates/removes
the output directory, opens the shared log and copies the resolved
configuration. All ranks synchronize before using those outputs.
``overwrite_output_dir`` retains its existing meaning; it is false in the
example. Even with overwrite disabled, log/config initialization is not an
append/resume API. Repeated ``build_env()`` is a no-op; explicit
``build_env(force=True)`` reinitializes the environment collectively.
Nonchief informational messages are suppressed by default and do not open
or truncate the shared log. Rank-local fatal errors reach the launcher
without waiting at an I/O error barrier, so a chief or worker failure
aborts the job instead of waiting forever for a different collective.

``bmtk.utils.io.ioutils.set_world_comm(comm)`` may configure the shared
communicator before ``Config.build_env()`` and model construction; I/O and
training use that same communicator. A duplicated communicator is tested.
Changing communicators after setup/training starts is unsupported, and
the launcher's failure handling still aborts the entire launched MPI job.
Without the launcher, serial I/O remains independent of optional ``mpi4py``.
An MPI configuration without the launcher, or a non-MPI configuration under
the launcher, raises before environment output setup. Plain
``mpiexec -n 2 python run_training.py`` is therefore not a substitute for
the early-device-selection launcher.

Support summary
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Surface
     - MPI backend scope
   * - Parallelism
     - One host; one complete network per process; equal sample batches
   * - Precision and learning
     - FP32, BPTT, one condition, zero state, no optimizer clipping
   * - Objective
     - Qualified sample-separable means, including TargetFiringRate
   * - Communication
     - Host MPI Scatter and gradient mean; not NCCL or CUDA-aware MPI
   * - Outputs and callbacks
     - Chief-owned environment and training outputs; collective control flow
   * - Restore
     - Explicit complete checkpoint plus collective synchronization
   * - Not admitted
     - Multi-node, mixed precision, local rules, combined training/inference

Implementation and validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The chief alone generates each global host batch. Numeric sample and target
leaves are scattered together into contiguous local slices, preserving Poisson
counts, input-generator seeds and sample identities. Each process computes a
local sample-mean gradient through the existing BPTT/checkpoint path.
Nonchief receive buffers are reused by leaf shape and dtype; chief in-place
Scatter uses its original sample slice without a redundant receive copy.
Eligible contiguous,
writable FP32 GPU inputs use a zero-copy **CPU** DLPack view before the required
local GPU upload; other dtypes retain ordinary TensorFlow conversion. Uploads
complete before these host buffers can be reused or changed. This avoids
repeated host snapshots without retaining caller-owned mutable storage on GPU.
The same completed-upload boundary is used for reusable reduced-gradient
buffers, preserving earlier GPU gradient snapshots across later updates.
With one process, finite canonical tensor gradients remain on their original
device; no host gradient download, mean reduction or re-upload is needed.
Canonical dense FP32 gradients from all trainable surfaces are packed into
one host MPI sum reduction per update and divided by process count before
local Adam or another admitted optimizer runs. This is not a per-timestep
collective, CUDA-aware MPI or neuron exchange. Globally averaged scalar metrics
use a separate reduction.

The eager ``TrainingEngine.train_step`` boundary coordinates compiled local
gradient and optimizer functions; MPI communication is outside TensorFlow
graphs. The high-level ``rnn.train()`` path uses it automatically. The private
traced ``_distributed_train_step`` is not an MPI entry point. Optimizer
constraints and recurrent/named-input shadow refresh remain in the ordinary
update lifecycle. Internal Poisson streams fold in the process rank and retain
checkpoint replay; changing process count does not reproduce identical noise.

Startup broadcasts chief model variables, **all** canonical recurrent and
named-input masters (including frozen inputs), optimizer slots and noise
variables. After explicitly restoring a chief checkpoint, every process must
collectively call ``rnn.data_parallel.synchronize(rnn.training_engine)`` before
another update. Checkpoints must still enumerate all intended masters: this
does not add an automatic generic resume API or repair an incomplete checkpoint.

Input generation and training callbacks run only on the chief. Training loss
metrics and early-stop decisions are synchronized so all ranks take the same
number of updates. Custom callbacks must respect this ownership: mutating
chief weights/optimizer/noise without a collective synchronization is not
supported. Uncaught worker errors are printed and abort the MPI job rather
than leaving peers blocked. Tracebacks and pending standard output/error are
flushed before aborting, since MPI abort bypasses Python's normal shutdown.
Run all ranks through the launcher and maintain identical collective control flow.

The initial FP32/BPTT/one-condition/zero-state/equal-full-batch/no-clipping and
sample-separable-loss restrictions above still apply. Dense canonical
gradients and numeric or Boolean sample leaves are required. Mixed precision,
stateful online population-rate updates, local rules and multi-node execution
need separate qualification. Combined MPI training/inference in ``rnn.run()``
is rejected to prevent duplicate inference files; run inference separately.

Contributor tests and measured scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DPointNet MPI smoke`` CI workflow installs the optional MPI dependency
and runs real CPU subprocesses in Keras 2 and Keras 3 environments, in addition
to mocked-communicator unit tests. From the checkout root:

.. code-block:: bash

   BMTK_DPOINTNET_DISABLE_FUSED_CUDA=1 BMTK_RUN_MPI_TESTS=1 \
     python -m pytest -q tests/simulator/dpointnet/test_mpi_integration.py
   mpirun -n 2 python -m bmtk.utils.dpointnet_mpi --cpu \
     tests/simulator/dpointnet/process_parallel_probe.py --output /tmp/mpi-proof.json

The integration tests cover ordinary serial I/O, one/two MPI ranks, configured
communicators, overwrite/no-overwrite, repeated/forced environment setup,
chief directory/config-copy errors, worker fatal logging and the complete
JSON-to-training example with exactly four optimizer updates and finite
loss/weight outputs. Error tests have external time limits and reject timeout
as success. The numerical oracle independently compares gradients/Adam updates,
constraints, state, explicit restore, callback/input ownership and Poisson replay.
CPU smoke tests do not qualify physical GPU placement, native kernels or speed.

The pre-integration V10032 qualification used 66,658 neurons, 24,450,554 recurrent
edges, 500 timesteps, FP32/NEST, checkpoint interval 25 and recurrent-only Adam.
Twenty synchronized full-output/diagnostic-write samples after three excluded
warmups measured 3.686884 s for the efficient single-GPU batch-16 reference
and 3.919032 s for two processes at local 16/global 32: **1.881528x throughput**.
This is not a universal speed guarantee, single-GPU batch-32 comparison,
convergence result or a new benchmark of subsequent I/O/documentation changes.
The earlier GTX1080Ti **1.953x** prototype used a different, counts-only loop
and must not be treated as a matched hardware comparison.

Alternative: single-process MirroredStrategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose ``run.distribution.strategy="mirrored"`` explicitly to run both GPUs
inside one TensorFlow process. This is an alternative, not the recommended
starting point for the qualified workload above. Both it and MPI remain opt-in.
Do not launch this configuration once per MPI rank.

This **configuration fragment**, not a standalone experiment, selects two
visible GPU ordinals:

.. code-block:: json

   {
     "run": {
       "dtype": "float32",
       "batch_size": 16,
       "distribution": {
         "strategy": "mirrored",
         "devices": ["/GPU:0", "/GPU:1"],
         "global_batch_size": 32
       }
     },
     "training": {
       "training_approach": "single",
       "initial_state": {"module": "zero_state"}
     }
   }

Run the ordinary Python entry point once, without ``mpirun`` or ``mpiexec``:

.. code-block:: bash

   python run_dpointnet.py config.mirrored.json

If ``devices`` is omitted, MirroredStrategy selects its local devices.
The shared FP32/BPTT/loss/batch/initial-state restrictions above still apply.
It uses TensorFlow's gradient collectives: canonical gradients are restored
locally, and the local scalar mean is divided by replica count before
differentiation. The optimizer's sum reduction then produces the global mean
gradient. Training and validation metrics are reduced before the single host
callback.

Native CSR metadata is owned per selected device, and native admission checks
the selected homogeneous GPU architectures against both operator manifests.
The fixed-queue CSR routing arrays are also resident on each worker.
Immutable cell tensors used by timestep kernels (including scalar timestep,
threshold and surrogate/gradient gains) are materialized as worker-local eager
captures before temporal/checkpoint graphs are traced. A local identity inside
a checkpoint graph still fetches its source at every chunk boundary.
The rollout frame is nested and thread-local, not a global cell-value cache.
Assignable coefficient variables remain live resources; replacing immutable
tensors requires ordinary TensorFlow retracing.
Localize scalar dependencies as well as large arrays: a four-byte peer fetch
on every timestep can serialize otherwise independent GPU streams.
The live type-identity dispatch predicate is copied to the host once per
rollout, rather than read back from the GPU at every forward/recomputed/VJP
step. Coefficient identity is still rechecked and incompatible live values
retain the existing fallback or explicit-required error.
