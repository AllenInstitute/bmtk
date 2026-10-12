Pascal GPU compatibility
========================

Pascal includes GTX1080Ti and Titan Xp GPUs (compute capability 6.1).
Read this page only if deploying on those devices.

.. warning::

   Architecture coverage alone is not workload qualification. Automatic routing
   now admits compatible Pascal CSR currents and repaired NEST state kernels,
   but initialization, training and inference still need measured VRAM headroom
   for your batch, connectivity and inputs.

Before choosing a Pascal GPU
----------------------------

* Confirm that TensorFlow, its CUDA dependencies and the driver support the
  device. A CUDA driver version alone is not enough.
* If using custom operators, include architecture 61 explicitly and verify
  both operator manifests; see :doc:`dpointnet_performance`. It is absent
  from the implicit build target list.
* Keep automatic acceleration rather than forcing packed flags from a newer
  GPU. It selects SM61 CSR currents, active queues and eligible native
  accumulation, while retaining canonical-gradient boundaries.
* Rebuild both operators from this source. Generic NEST launches now use
  kernel-specific occupancy limits. Automatic state admission checks a
  defaulted ``launch_geometry_version`` capability attribute; an old unmarked
  library is not admitted even if its manifest advertises SM61.
* Check the actual workload's initialization and update memory. Titan Xp
  has more memory than GTX1080Ti; fitting one does not establish fitting both.
* Test the configured dynamics, precision, inputs, updates and restoration.
  Do not change those scientific choices just to make a compatibility test pass.

NEST-compatible dynamics remain the default. Explicit legacy dynamics are
available for legacy reproduction, not as a scientifically equivalent
workaround for a NEST-compatible-mode device failure.

Why the native current route matters
------------------------------------

Unfused input projection can materialize large temporary connection tensors.
In the measured 66,658-neuron batch-10 V1 inference, the old Pascal fallback
exceeded memory headroom for hard reset and ran out of memory for soft reset.
With the existing SM61 native CSR current route, the same paired protocol
passed on GTX1080Ti at 5.16/5.18 GiB peak TensorFlow allocation. Do not extrapolate
this to batch-32 BPTT, whose temporal tape and optimizer add memory.

cuDNN compatibility is a separate surface: some convolution shapes are not
supported on Pascal by recent cuDNN. The synchronization-loss regression uses
a CPU-placed independent convolution oracle while retaining the actual
DPointNet value/gradient calculation on GPU and unchanged strict tolerances.
This does not install a general CPU fallback in the model.

Measured qualification
----------------------

On GTX1080Ti with TensorFlow 2.21/CUDA 12.9 and both rebuilt cluster-fat
operators, the full DPointNet run passed 2,424 cases and skipped 29.
Its only failure was the independent convolution reference VJP being placed
on GPU after a CPU forward. The corrected reference ran on CPU; a scoped
25-case recovery passed, explicitly asserting that the actual DPointNet
loss and gradient stayed on GPU. This provides 2,425 unique passing cases
with the original failure receipt preserved, not a second full green run.

The actual 66,658-neuron V1 network then passed three accepted updates each
with hard-surrogate and soft-reset training at batch eight, 500 steps and
25-step checkpoints. Both used automatically selected native currents,
state/event VJPs and recurrent accumulation, checked finite updates and
constraints, and strictly restored model and optimizer state.
Peak TensorFlow allocation was 6.12 GiB for hard and 6.04 GiB for soft.
The two post-trace updates took about 5.8 seconds each; these are startup
measurements, not a twenty-sample full-training benchmark or convergence study.

A separate four-workload CSR forward/VJP benchmark used three excluded
warmups and twenty synchronized samples for each route, including canonical
gradient restoration. Winners varied with batch and activity; isolated
packed results do not establish a universally fastest full-training route.
Packed Pascal flags therefore remain explicit, not automatically forced.
Batch-32 V1 training on an 11-GiB device is not qualified by these results.

Visual inputs
-------------

DPointNet includes a pre-Volta GPU path for LGN spatial filtering that avoids
unsupported cuDNN convolution. It retains the original padding, normalization,
row order and temporal filters, and chunks large movies.
This is specific to LGN filtering, not a general convolution fallback.
Measure actual input shapes and memory before using it in a model.

If execution fails
------------------

Retain the first exception, model/config revision, precision report, operator
manifests, TensorFlow/CUDA versions and GPU name. Establish whether it is a
missing target, unsupported library operation, launch-resource or memory
failure before selecting a recovery.
Initial-state recovery notices are explained in :doc:`dpointnet_input_recovery`.

A successful build or another GPU family's test suite is not a complete
Pascal qualification. Kernel implementation details belong in
:doc:`dpointnet_development`, not in the ordinary configuration workflow.
