#########
DPointNet
#########

DPointNet is a BMTK engine for the training and simulation of large-scale biorealistic neuronal circuits 
utilizing deep-learning techniques. Users can take a novel or existing network and train and save synpatic 
weights using pre-determined inputs and expected outputs. It can also be used for purely inference/simulation
of large scale networks often in a way that is much faster than other simulators.

For more information see paper on using software for building and simulation of cortical circuit model:

`Ito et. al., 2026 <https://www.biorxiv.org/content/10.64898/2026.03.13.711751v1>`_



Installation
============

DPointNet must be installed using the base bmtk package as seen in :doc:`the instllation guid <installation>`.

Besides the base dependencies, it requires tensorflow 2.14 to 2.16, and may work with gpu or without. To install
with gpu run the following in your environment:

::

    $ pip install tensorflow[and-cuda]

or without a gpu:

::

    $ pip install tensorflow

Optional fused CUDA operator
----------------------------

DPointNet can use a fused CUDA operator for recurrent and input synaptic currents. This optional operator
requires an NVIDIA CUDA toolkit with ``nvcc``, a C++17 compiler, and a GPU-enabled TensorFlow installation.
Build it from the same environment in which BMTK and TensorFlow are installed:

::

  $ python -m bmtk.simulator.dpointnet.custom_ops.build

This module form ensures that the build uses the active Python environment and does not require a console script on
``PATH``. Installing this version of BMTK also creates ``bmtk-build-dpointnet-cuda`` in the environment's executable
directory; the shortcut is available when that environment is activated.

The build targets compute capabilities 7.0, 7.5, 8.0, 8.6, 8.9, 9.0, and 12.0 by default, with
``compute_120`` PTX. To build for a different set of architectures, provide space-separated architecture numbers:

::

  $ DPOINTNET_CUDA_ARCHS="80 86 90" python -m bmtk.simulator.dpointnet.custom_ops.build

The operator requires exactly one visible GPU. Set ``use_fused_cuda`` to ``true`` in ``rnn_cell_params`` to
require the operator, or to ``"auto"`` to use it when available and otherwise fall back to TensorFlow. The
default is ``false``. Rebuild the operator after changing TensorFlow or CUDA installations or after updating
BMTK custom-op source; SavedModel graphs containing an older custom-op signature must also be regenerated.

Recurrent backward-kernel selection is controlled separately by ``use_pair_projection``:

* ``"auto"`` (default) selects the pair-projected kernel when fused CUDA is active, the configured batch size
  is 32, and the synaptic basis has four columns. Other configurations use the general fused backward kernel.
* ``true`` explicitly enables pair projection for any positive configured batch size and basis width with
  fused CUDA. Shapes outside the batch-32 specialization use a general projection kernel and the general
  backward reduction. This opt-in path does not require SM86 or packed backward.
* ``false`` always uses the general fused backward kernel and avoids compact-pair metadata construction.

For example:

::

  "rnn_cell_params": {
    "use_fused_cuda": "auto",
    "use_pair_projection": "auto"
  }

The pair-projected kernel computes each distinct postsynaptic-neuron/synapse-type basis projection once and
reuses it across recurrent edges. It only changes the recurrent backward pass; inference and learning rules
that do not differentiate through the recurrent dynamics do not benefit. Master weights and checkpoints remain
in canonical edge order.

On SM86 or newer GPUs, ``use_packed_sm120_backward="auto"`` additionally selects a packed FP32 recurrent
backward for float16 batch-32 models with four basis columns and ``uint32`` compact-pair metadata. Set it to
``false`` for a same-GPU comparison with the prior pair-projected kernel, or to ``true`` to require the packed
path and fail when any prerequisite is absent. It defaults to ``"auto"``; SM80 and older retain the prior
pair-projected kernel. The option name is retained for configuration compatibility after qualification on SM86.
The default CUDA build includes native SM86, SM89, and SM120 code plus ``compute_120`` PTX.

``use_packed_sm120_external_backward`` controls the corresponding weight-only backward for named input
populations. It has the same ``"auto"``, ``true``, and ``false`` selection contract and additionally builds
compact pair metadata only for trainable input weights. Fixed LGN connectivity therefore retains its smaller
metadata layout. The packed kernel splits sparse source rows across blocks, returns no external-activity
gradient, and writes weight gradients through the CSR-to-canonical edge map. Fixed input populations skip both
backward kernels. SM80 and older and incompatible shapes retain the existing external backward.

For batch 32 with four basis columns, fused recurrent and spike-input forward passes group each source row's
active batch samples into a 32-bit mask and launch one CUDA block per active source row. This avoids launching
one block for every batch/source combination when biological spike tensors are sparse; other shapes retain the
general forward kernel.

Set ``use_active_row_forward=true`` to opt into that source-row compaction for batch sizes 1 through 32
with four basis columns. The mask reads only actual samples; it does not pad the physical batch or discard
simulations. The default remains ``false`` (the existing automatic batch-32 behavior is unchanged).
Half-precision scatter addition order and rounding differ from the general forward path, so numerical
and task-level validation is required; bitwise spike-trajectory identity is not promised.

Set ``use_small_batch_recurrent_backward=true`` to opt into a batch-1-through-8 recurrent backward kernel.
One block traverses each presynaptic row, reduces sample contributions locally, and writes each FP32
weight gradient once instead of atomically combining separate sample blocks. It supports general basis
widths and can be combined independently with explicit pair projection and CSR gradient output. It is
disabled by default: correct small-tensor behavior does not guarantee better full-network performance.

Set ``use_direct_csr_recurrent_gradient=true`` to accumulate dynamics weight gradients in contiguous CSR
order, then restore the existing master-variable order once at the rollout boundary. This now supports
general fused backward kernels, float16/float32, and general batch/basis sizes; it requires individually
trainable recurrent edges but not packed backward or SM86. Segmented BPTT uses its existing final
gradient transform. Full BPTT uses a retained tape without recomputing the forward rollout. Only the
rollout's gradients are transformed: outer weight-regularizer gradients remain in master-variable order
and are combined after restoration. Optimizer variables, slots, and constraints retain their ordering.

These paths are independent opt-ins, not a change to batch composition, learning rate, loss definitions,
sequential/parallel update semantics, or stopping criteria. On a short tuned V1 batch-5 full-BPTT series
benchmark (RTX 3090, 500 steps, all nine paper losses), active-row forwarding plus pair projection
measured 5.40 s per two-update step versus 7.41--7.52 s baseline. Direct CSR alone was approximately
neutral (7.44 s) and used more memory; small-batch backward was slower (10.98 s, or 8.49 s with direct
CSR). These are short-run results, not full-training convergence or other-architecture qualification.

Activating the measured batch-5 speedup
-------------------------------------

Merge the following settings into an existing paper training configuration, retaining its inputs,
losses, initialization and callbacks. This is an opt-in execution profile, not a new learning-rate
or stopping policy. At batch 5, ``use_pair_projection="auto"`` does **not** enable projection;
use the JSON boolean ``true`` explicitly, together with ``use_active_row_forward=true``.

.. code-block:: json

  {
    "run": {
      "batch_size": 5,
      "seq_len": 500,
      "dt": 1.0,
      "dtype": "float16"
    },
    "rnn_cell_params": {
      "use_fused_cuda": true,
      "use_fused_state": true,
      "use_active_row_forward": true,
      "use_pair_projection": true,
      "use_packed_sm120_backward": false,
      "use_packed_sm120_external_backward": false,
      "use_small_batch_recurrent_backward": false,
      "use_direct_csr_recurrent_gradient": false,
      "use_fixed4_input_forward": false,
      "use_fused_current_accumulation": false,
      "track_voltage_penalty": false,
      "return_voltage_sequences": true
    },
    "training": {
      "training_approach": "series",
      "n_epochs": 75,
      "steps_per_epoch": 25,
      "gradient_checkpointing": false,
      "pack_spike_checkpoints": false,
      "optimizer": {"name": "exp_adam", "epsilon": 1e-11},
      "learning_rate": {"schedule": "none", "learning_rate": 0.005}
    }
  }

If individual training parameters specify ``batch_size``, keep each at 5 as well. With evoked
and spontaneous parameters, series mode still performs two sequential optimizer updates per
outer step: 3,750 updates over 75 x 25 steps. There is no batch padding or joint-condition update.
Keep offline voltage loss in both conditions (``online=false`` or omit ``online``).

Rebuild the CUDA operators from the updated checkout in the same environment used to train.
For the measured RTX 3090 configuration:

::

  $ DPOINTNET_CUDA_ARCHS="86" python -m bmtk.simulator.dpointnet.custom_ops.build
  $ python -c "import bmtk; print(bmtk.__file__)"

Choose the architecture for the actual GPU, and verify that the printed import path is the intended
checkout. Existing installations do not pick up an uninstalled source checkout automatically.
The packed-kernel flags above are disabled because batch 5 cannot use those specializations.
General direct CSR and the experimental small-batch reduction are also disabled because they did
not improve this full-network benchmark. Library defaults remain unchanged; an unmodified paper
configuration will not automatically enable the two new opt-ins. A 27--28% reduction in the measured
training-step time is not a measured 27--28% reduction in complete 75-epoch job time. Validate the
long-run behavior and hardware of interest before adopting this profile for production.

SONATA weight export restores the source edge-row order after runtime recurrent sorting. Interleaved
edge groups use ``edge_group_index`` to align properties with population rows, and multiple edge
populations retain separate input offsets. Physical weight export factors are applied before restoring
row order. The exporter still writes its existing single weight group; this is not a byte-for-byte copy
of every input HDF5 attribute, group structure, or index dataset.

Set ``use_fixed4_input_forward=true`` to select a fixed-four gather forward for input populations with exactly
four incoming edges per postsynaptic neuron. One thread owns each ``(batch, post)`` output and writes all four
basis values without scatter atomics. Selection is derived from connectivity structure rather than population
name; nonqualifying populations retain grouped/general forwarding. The default is ``false`` to preserve prior
mixed-precision summation semantics. Backward continues to use the source-CSR path, preserving canonical
trainable-weight gradients and the no-activity-gradient contract.

Set ``use_fused_current_accumulation=true`` to thread recurrent and fused spike-input currents through one
additive CUDA buffer instead of materializing each source and combining them with a separate ``AddN``. The
option defaults to ``false`` and requires fused CUDA currents. Its custom gradient passes the upstream current
gradient unchanged through the accumulator while retaining independent canonical gradients for every trainable
weight surface. Current-type or otherwise non-fused inputs retain the existing TensorFlow addition path.

The optimization exchanges startup time and a small amount of persistent GPU memory for faster batch-32 BPTT.
Measured examples include a 55.7% update-time reduction on a 66,658-neuron network on A100-PCIE-40GB and a 37.9%
reduction on a 19,570-neuron network on RTX 3090. The corresponding peak-memory increases were 0.54% and 0.09%.
These measurements are hardware- and topology-dependent; benchmark representative training before forcing the
pair kernel. Small-batch specializations were tested but regressed the complete networks, so ``"auto"`` does not
select pair projection below batch 32.

The optional ``use_fused_state`` cell parameter fuses the GLIF membrane,
refractory, ASC, PSC, spike, and delayed-history transition. It is ``false`` by
default. ``"auto"`` selects it only when the CUDA library is available, the
synaptic basis has four columns, and the triangular surrogate is active;
``true`` requires those conditions and otherwise raises a configuration error.
Gaussian surrogate models retain the TensorFlow transition. Soft reset avoids
retaining refractory history in the custom backward; hard reset remains
supported and tested.


Overview
========

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
      

  .. tab-item:: Python

    .. code:: python





DPointNet Initialization
========================

Setting up a simulation environment/workspace
---------------------------------------------

Initializing DPointNet instance
-------------------------------

First we must instantiate a DPointNet simulator instance, which passing in parameters 
required to build and run the model. Some of these parameters may be changed later on.

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json

        {
          "target_simulator": "DPointNet",

          "run": {
            "seq_len": 500,
            "dt": 1.0,
            "default_seed": 3000,
            "batch_size": 10,
            "train_recurrent_weights": true,
            "dtype": "float32",
            "single_gpu_strategy": "one_device"
          }
        }
      

  .. tab-item:: Python

    .. code:: python

        from bmtk.simulator import dpointnet

        rnn = dpointnet.RNN(
            seqlen=500.0, 
            dt=1.0,
            default_seed=3000,
            batch_size=10,
            dtype="float32",
            train_recurrent_weights=True,
            single_gpu_strategy="one_device",
        )


.. dropdown:: Available "run" options
  :open:

    .. list-table::
        :header-rows: 1

        * - option
          - description
          - default
        * - seq_len
          - The number of time steps that will be used in training and inference 
          - 
        * - dt
          - The time interval, in milliseconds, for each sequence step
          - 1.0
        * - default_seed
          - The default RNG seed to use when building the model and any of DPointNet functions (like spike generators or training) - when not explicity stated.
          - 
        * - batch_size
          - The default number of simulataneous batches for processing during feed-forward input into the RNN. May be overridden for training and inference.
          - 1



Setting hyper-parameters for training/inference
-----------------------------------------------

Next we must set hyper-parameters that are used by the back-end deep-learning model. You must first specify the `cell_model` which takes care of reproducing
the internal simulation output plus rules for determining gradient calculations. Current DPointNet only supports `GLIF3Cell` model that reproduces 
the `GLIF point-neuron models <https://brain-map.org/our-research/computational-modeling/glif-single-neuron-models>`_.

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json

        {
          "rnn_cell_params": {
            "cell_model": "GLIF3Cell",
            "<parameter_1>": <value_1>,
            "<parameter_2>": <value_2>,
            "<parameter_3>": <value_3>,
            ...
          }
        }
      

  .. tab-item:: Python

    .. code:: python


.. dropdown:: Available "rnn_cell_params" options

    .. tab-set::

        .. tab-item:: GLIF3Cell

            .. list-table::
                :header-rows: 1

                * - option
                  - description
                  - default
                * - gauss_std
                  - 
                  - 0.5
                * - dampening_factor
                  - Scale applied to the spike surrogate derivative.
                  - 0.3
                * - recurrent_dampening_factor
                  - Retained recurrent temporal-gradient multiplier. ``0.0`` blocks this gradient and ``1.0`` leaves it undampened.
                  - 0.5
                * - voltage_gradient_dampening
                  - Retained multiplier for the membrane-voltage self-loop gradient. Synaptic-current gradients are not scaled.
                  - 0.5
                * - recurrent_weight_scale
                  - 
                  - 1.0
                * - lr_scale
                  - 
                  - 1.0
                * - max_delay
                  - 
                  - 5
                * - pseudo_gauss
                  - 
                  - False
                * - train_recurrent
                  - 
                  - True
                * - train_recurrent_per_type
                  - 
                  - True
                * - noise_seed
                  - 
                  - 0
                * - hard_reset
                  - 
                  - False
                * - tau_basis
                  - 
                  - <None>
                * - synaptic_basis_weights
                  -
                  - <None>
                * - use_fused_cuda
                  - Use the optional fused CUDA synaptic-current operator. ``True`` requires it; ``"auto"`` falls back to TensorFlow when unavailable.
                  - False
                * - use_pair_projection
                  - Select the recurrent CUDA backward kernel. ``"auto"`` uses pair projection for batch 32 with four basis columns; ``True`` requires it; ``False`` forces the general kernel.
                  - "auto"
                * - use_packed_sm120_backward
                  - Select the packed recurrent backward on SM86 or newer. ``True`` requires float16, batch 32, four basis columns, ``uint32`` compact-pair metadata, and qualified hardware; ``False`` retains the prior pair kernel.
                  - "auto"
                * - use_packed_sm120_external_backward
                  - Select the packed weight-only backward for trainable input populations on SM86 or newer. Fixed inputs build no pair metadata or backward; ``False`` retains the prior external kernel.
                  - "auto"
                * - use_fixed4_input_forward
                  - Use the one-owner input forward for populations with exactly four incoming edges per postsynaptic neuron. Nonqualifying populations retain grouped/general forwarding.
                  - False
                * - use_fused_current_accumulation
                  - Accumulate recurrent and fused spike-input currents through one additive CUDA buffer. Requires fused CUDA currents; current-type inputs retain the TensorFlow addition path.
                  - False
                * - use_direct_csr_recurrent_gradient
                  - Accumulate recurrent gradients in CSR order and restore master-variable order at the full or segmented BPTT boundary. Requires individually trainable recurrent edges and fused CUDA; packed kernels are optional.
                  - False
                * - use_small_batch_recurrent_backward
                  - Experimental local batch reduction for batch sizes 1 through 8 with fused CUDA; independent of pair projection and gradient layout.
                  - False
                * - use_active_row_forward
                  - Opt into active-source-row forwarding for batch sizes 1 through 32 with four basis columns and fused CUDA. Does not pad or change the training batch.
                  - False
                * - track_voltage_penalty
                  - Accumulate a compact neuron-mean voltage penalty at each timestep. Enable only with an online ``VoltageRegularization`` loss.
                  - False
                * - voltage_penalty_mode
                  - Compact voltage penalty: ``"range"`` penalizes voltages outside the normalized range [0, 1], while ``"threshold"`` penalizes distance from threshold.
                  - "range"
                * - return_voltage_sequences
                  - Return neuron-resolved voltage sequences. Setting this to ``False`` requires ``track_voltage_penalty=True``.
                  - True


Setting the Network Model
=========================

Before either training or inference can begin, DPointNet must have instantiated network files that describes the cells, 
synapses, and all the required properties. You can build a network from scratch using the :doc:`BMTK Network Builder <builder>`,
or pre-built models like the ones developed at the `Allen Institute <https://brain-map.org/our-research/computational-modelling>`_
or from other labs.


.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
      
        "networks": {
          "nodes": [
            {
              "nodes_file": "$NETWORK_DIR/glifs_nodes.h5",
              "node_types_file": "$NETWORK_DIR/glifs_node_types.csv"
            },
            {
              "nodes_file": "$NETWORK_DIR/virts_nodes.h5",
              "node_types_file": "$NETWORK_DIR/virts_node_types.csv"
            }
            ],
            "edges": [
            {
              "edges_file": "$NETWORK_DIR/glifs_glifs_edges.h5",
              "edge_types_file": "$NETWORK_DIR/glifs_glifs_edge_types.csv"
            },
            {
              "edges_file": "$NETWORK_DIR/virts_glifs_edges.h5",
              "edge_types_file": "$NETWORK_DIR/virts_glifs_edge_types.csv"
            }
          ]
        }

  .. tab-item:: Python

    .. code:: python

        import numpy




Network requirements
--------------------

instantiating the Network
-------------------------

Networking options
^^^^^^^^^^^^^^^^^^

Network Components
^^^^^^^^^^^^^^^^^^

Combining multiple networks together
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Filtering for subnetworks
^^^^^^^^^^^^^^^^^^^^^^^^^


Input Stimuli
=============

Spiking Stimulus
----------------

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
          "inputs": {
            "<INPUT_NAME1>": {
              "input": "spikes",
              "module": "<INPUT_MOD1>"
              "node_set": "<virtual_pop_1>",
              "<module_params_1>": <value_1>,
              "<module_params_2>": <value_2>,
              ...
            },
            "<INPUT_NAME2>": {
              "input": "spikes",
              "module": "<INPUT_MOD1>"
              "node_set": "<virtual_pop_1>",
              "<module_params_1>": <value_1>,
              "<module_params_2>": <value_2>,
              ...
            }
          }
        }
      

  .. tab-item:: Python

    TBA




Spike-inputs modules and options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. dropdown:: Available "run" options
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
        * - random
          - 
        * - bernoulli_spikes 
          -
        * - poisson_spikes
          - 
        * - lgn_tf
          - 
        * - spikes_files
          - 
        * - custom_spikes_functions
          - 


Building your own inputs module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




Initial Conditions
==================

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
            "initial_states": {
                "<NAME1>": {                    
                    "module": "<INIT_MOD1>",
                    "run_on": "all",
                    "<module_params_1>": <value_1>,
                    "<module_params_2>": <value_2>,
                    ...
                },
                "<NAME2>": {                    
                    "module": "<INIT_MOD2>",
                    "run_on": "epoch",
                    "<module_params_1>": <value_1>,
                    "<module_params_2>": <value_2>,
                    ...
                }
           }
        }
      

  .. tab-item:: Python

    .. code:: python



Available modules
^^^^^^^^^^^^^^^^^

.. dropdown:: Available "run" options
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
        * - zero_state
          - 
        * - random_state
          - 
        * - from_input
          - 
        * - cached_states
          - 


Creating your own module
^^^^^^^^^^^^^^^^^^^^^^^^






Training Options
================

Training hyper-parameters
-------------------------

The default training path remains full-sequence BPTT with neuron-resolved voltage
outputs. The performance and memory options below are conservative so existing
configurations retain their previous behavior:

.. list-table:: Training and output options
   :header-rows: 1

   * - option
     - default
     - description
   * - ``gradient_checkpointing``
     - ``False``
     - Enable segmented exact BPTT recomputation.
   * - ``gradient_checkpoint_chunk_size``
     - ``25``
     - Number of timesteps per recomputed chunk when checkpointing is enabled.
   * - ``pack_spike_checkpoints``
     - ``False``
     - Pack the binary delayed-spike state into positive 31-bit words at chunk boundaries.
   * - ``regenerate_initial_state_each_epoch``
     - ``True``
     - Generate fresh configured initial state at each epoch boundary. Set to ``False`` to reuse the initial state across epochs.
   * - ``learning_rule``
     - ``"bptt"``
     - Select standard BPTT or a registered local learning rule.

Segmented exact BPTT retains recurrent state only at temporal chunk boundaries
and recomputes each chunk during the backward pass. Internal Poisson timestep
progression is explicit recurrent state, so recomputation uses the same
stateless random draws as the original forward pass.

.. code:: json

    {
      "training": {
        "gradient_checkpointing": true,
        "gradient_checkpoint_chunk_size": 25,
        "pack_spike_checkpoints": true
      }
    }

``gradient_checkpoint_chunk_size`` must be between 1 and the configured sequence
length. Smaller chunks reduce activation memory but increase recomputation
overhead. The default is 25 timesteps; checkpointing remains disabled unless
explicitly requested.

``pack_spike_checkpoints`` applies only to the first recurrent state, which is
the binary delayed-spike history for the built-in GLIF cell. Every nonzero value
is treated as a spike. Voltage, refractory, ASC, PSC, and replay-safe Poisson
state remain unpacked. Packing is opt-in and has no effect unless segmented
checkpointing is enabled. It reduces checkpoint memory but can add bit-packing
work, so benchmark representative training before enabling it by default.

Compact online voltage regularization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full voltage sequences have shape ``[batch, time, neurons]`` and can dominate
output-gradient memory. If no analysis or learning rule needs neuron-resolved
voltages, DPointNet can instead emit one pre-aggregated penalty value per sample
and timestep:

.. code:: json

    {
      "rnn_cell_params": {
        "track_voltage_penalty": true,
        "voltage_penalty_mode": "range",
        "return_voltage_sequences": false
      },
      "training": {
        "parameters": [
          {
            "loss_functions": {
              "voltage": {
                "module": "VoltageRegularization",
                "penalty_mode": "range",
                "online": true
              }
            }
          }
        ]
      }
    }

The cell and loss ``penalty_mode`` values must match. Online mode supports
``"range"`` and ``"threshold"`` penalties and does not support a core mask.
The defaults are ``online=False``, ``track_voltage_penalty=False``, and
``return_voltage_sequences=True``. Keep those defaults when another consumer,
including a local learning rule, requires full voltages.

Callbacks
---------

Default Callbacks class
^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
          "callbacks": {
            "class": "Callbacks",
            "starting_epoch": 0,
            "callbacks_dir": "training_callbacks_intro_l4_overall_distribution",
            "verbose": "on_step",
            "epoch_store_weights": "latest",
            "epoch_cache_weights": false,
            "sonata_output_dir": "network.trained_weights.best",
            "losses_table_csv": "losses.csv",
            "performance_table_csv": "performance.csv"
          }
        }
      

  .. tab-item:: Python

    .. code:: python




.. dropdown:: Available "Callback" options
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
          - default
        * - callbacks_dir
          - 
          - callbacks_outputs
        * - starting_epoch
          - 
          - 0
        * - verbose
          -
          - full
        * - time_fmt
          -
          - '%d-%m-%Y %H:%M'
        * - epoch_cache_weights
          -
          - False
        * - epoch_store_weights
          -
          - best
        * - sonata_output_dir
          -
          - trained_weights
        * - losses_table_csv
          -
          - losses.csv 
        * - performance_table_csv
          -
          - performance.csv


Building your own Callbacks class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Parameters
----------


Training Inputs
---------------


Initial State
-------------


Loss Functions
--------------

.. dropdown:: Built-in Loss Modules
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
        * - SpikeRateDistributionTarget
          - 
        * - TargetFiringRate
          - 
        * - OrientationSelectivityLoss
          -
        * - VoltageRegularization
          -
        * - SynchronizationLoss
          -
        * - EMDWeightRegularization
          -






Training Output
---------------




Running Inference
=================


Running an Inference
--------------------


Results/Output
--------------

.. tab-set::

  .. tab-item:: SONATA

    .. code:: json
        
        {
            "output": {
                "output_dir": "$OUTPUT_DIR",
                "log_file": "$OUTPUT_DIR/log.txt",
                "log_level": "INFO",
                "spikes_file": "$OUTPUT_DIR/spikes.h5",
                "overwrite_results": true
            }
        }
      

  .. tab-item:: Python

    .. code:: python







