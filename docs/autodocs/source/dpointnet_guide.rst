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
                  - 
                  - 0.3
                * - recurrent_dampening_factor
                  - 
                  - 0.5
                * - voltage_gradient_dampening
                  - 
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

    .. code:: python




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







