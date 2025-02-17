#####################################
Running Network Simulations with BMTK
#####################################

.. toctree::
    :hidden:
    :maxdepth: 2

    bionet
    pointnet
    filternet
    popnet


In this section we will show how to use BMTK and SONATA to run simulation(s) on a brain network model. 

Unlike other neural simulation tools which will create and simulate a network in one script, BMTK workflow design is to
to split up these two parts of the process. So before we can run a simulation we must first procure a network model 
(typically store using `SONATA circuit format <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#representing-circuits>`_). 
We can either download an existing model, or alternatively use a tool like the BMTK Network Builder to create one from 
scratch.

Once we have our network to run simulations on, we typically need to complete the following steps:

1. **Setup the enviornment**. For most networks, this entails downloading or creating any auxiliary files required for 
   network instantiation (morphology files, cell parameters, mod files, etc) and inputs (spikes, electrodes).

2. **Setup the SONATA configuration file(s)**. We use a json based SONATA configuration file to instruct BMTK how to 
   instiante a network (including location of circuits and any auxiliary files), run-time parameters, and what stimuli
   to use as input and what variables to record for our output. We can create a SONATA config file from scratch, or 
   download and edit an existing one using any prefered text-editor.

3. **Run the simulation(s)**. The majority of BMTK simulation can be fully defined in the SONATA config (although for 
   advanced users BMTK allows extensive customization using Python). Thus we just need to execute an pre-generated 
   python script with the above SONATA config file and let our simulation finish.
   
Once the simulation has completed it will automatically generate and save the results as specified in the SONATA 
configuration file. Although BMTK can run network models of different levels-of-resolutions, this is abstracted from 
the user, and it will use the appropiate underlying simulator library, e.g. **Simulation Engine**, depending on the cell models.
So no matter if the network is run using NEURON, NEST, DiPDE, or any other engine; the expected inputs and outputs 
are the same format

.. figure:: _static/images/bmtk-workflow-v2-simulation-highlighted.png
    :scale: 60%


The rest of this guide will go through each of the above steps in detail. 

.. card:: example network
  :class-card: .user-guide-example sd-border-2

  To help make the concepts for concrete we will also be referencing the **example** network simulation found 
  `here <https://github.com/kaeldai/bmtk/tree/refactor/documentation/docs/autodocs/source/example_bionetwork>`_. 
  This is a biophysically detailed network containing a 450 cells with a mixture of both multicompartment biophysically 
  detailed cells with parameters and morphologies download from the Allen Cell Types Database, surrounded by a ring of 
  point integrate and fire neuron. For the main configuration, feedforward synaptic stimuli will emerge from virtual 
  cells firing at a randomized rate.


1. Setting up the Environment
=============================

First step is to download and/or create neccesary files required to instantiate network and execute the simulation. At 
miniumum, we require the SONATA circuit file(s), simulation configuration, and a BMTK run script. In addition, we may also
need the followings depending on the type of model and simulation:

.. grid:: 1 2 2 2

  .. grid-item::
    :columns: 12 6 6 6

    * template files used to build the cell or synapse models (*Hoc Templates*, *NeuroML*, *NESTML*)
    * cell and synaptic dynamics attribute values,
    * cell morphologies (*SWC*, *Neuralucdia*),
    * simulation input and stimuli (*spike-trains*, *current wave-form*, *movie and auditory files*),
    * NEURON .mod files.

    We can put these files wherever we want as long as they are accessable during simulation execution. Although best 
    practices is to put them inside a single directory with the following structure.

  .. grid-item::
    :columns: 12 6 6 6

    .. figure:: _static/images/bmtk_sim_env.2024.png
        :scale: 40%


BMTK includes the `create_environment <SIMSETUP>`_ tool that can help new users generate an environmental directory from 
scratch. Another option we recommend, especially if running simulations on an already existing model, is to download an 
existing simulation environment and make changes as necessary.



.. card:: example network
  :class-card: .user-guide-example sd-border-2

  When creating the `BioNet example <https://github.com/kaeldai/bmtk/tree/refactor/documentation/docs/autodocs/source/example_bionetwork>`_ 
  we used the `build_network.py` python script to build and save the network model 
  into the **network/** sub-directory (see `BMTK Builder Guide <builder.html>`_ for more information on that process). With the network
  built we then used the following command to generate baseline strucutre plus `config.simulation.json` configuration and
  the `run_bionet.py` script used to execute the simulation:


  .. code:: bash

      $ python -m bmtk.utils.create_environment         \
                      --config-file config.iclamp.json  \
                      --overwrite                       \
                      --network-dir network             \
                      --output-dir output_iclamp        \
                      --tstop 3000.0                    \
                      --dt 0.1                          \
                      --report-vars v                   \
                      --iclamp 0.150,500,2000           \
                      --compile-mechanisms              \
                      bionet .
              
  This script will create the **components/** directory to place any auxiliary files for network instiation, but unless
  explicity defined, the corresponding subfolders will be empty. In particular our model's various cell-types require
  SWC morphology and dynamics parameters files that can be downloaded from the Allen Cell Types Database, renamed and 
  placed into their corresponding folders.

  .. figure:: _static/images/ctdb2bmtk_model_download.png
    :scale: 80%

  For simulation input our network will be stimulated by feed-forward pre-generated spike-trains that will be saved into 
  the **inputs/** folder using the `create_inputs.py` script

  .. code:: bash

    $ python create_inputs.py




2. Setting up Sonata Config file(s)
===================================

The primary interface thorugh which most users will run a simulation is through the **SONATA confiugration** file(s). 
Each simulation will have its own json config file that can be opened and modified by any text editor, allowing users
to readily adjust simulation, network, input, and output of any given simulation without any required coding or having
to learn the API for a specific simulator.

The config file is segmented into sections for the various aspects of running a full simulation. We will go into depth
of each section below.


"run"
^^^^^

The "run" section allows us set run-time parameters for the simulation. At minimum this includes 
the **tstop** parameter that determines the time step (in ms) when the simulation will stop. Other options, including
**tstart** (time at start of simulation, ms) and **dt** (time step interval, ms), may or may not be optional or even
used depending on the simulation.

The following attributes can be used by BMTK to set to time course of a given simulation.

.. dropdown:: "run" simulation attributes
  :open:

   .. list-table::
      :header-rows: 1

      * - name
        - description
        - required
      * - tstart
        - Start time of simulation in ms (default 0.0)
        - False
      * - tstop
        - Stop time of simulation (default 0.0)
        - True
      * - dt
        - The time step of a simulation; ms
        - True
      * - dL
        - For networks with morphological models, is a global parameter used to indicate to the simulator the maximum length 
          of each compartmental segment, in microns. If "dL" parameter is explicitly defined for a specific cell or cell-type,
          then BMTK will default to the more grainular option.
        - False
      * - spike_threshold
        - For networks using conductance based model, is a global paramter to indicate the threshold in mV at which a cell undergoes
          an action potential. Will not apply to integrate-and-fire type models. Value will be overwritten for any cell or cell-type
          that explicity defines "spike_threshold" parameter in network attribute.
        - False
      * - nsteps_block
        - Used by certain input/report modules to indicate time processing of timestep in blocks of every n time-steps. In particular
          for recording of spike-trains, membrane variables, or extraceullar potential; tells the simulation when to flush data 
          out of memory and onto the disk. 
        - False

.. card:: example network
  :class-card: .user-guide-example sd-border-2
  
  The "run" section for our example simulation contains the following:
  
  .. code:: json
    
      
      "run": {
        "tstop": 3000.0,
        "dt": 0.1,
        "dL": 20.0,
        "spike_threshold": -15,
        "nsteps_block": 5000
      },

  This will tell our simulation to run for 3,000 ms (3 seconds) with 0.1 ms timesteps, block process all the data every
  5000 steps (eg. 500 ms). The "dL" makes sure that for morphologically detailed cells each branch segment is no more 
  than 20 microns in lenght. And to record a spike when a cell reaches threshold of -15.0 mV.




"inputs"
^^^^^^^^

The "inputs" section of the SONATA config file is used to specify stimlus to apply to the network. It will contain one 
or more independent stimuli blocks of the format 

.. code:: json 

    {
        "<BLOCK_NAME>": {
            "input_type": "<STIMULI_TYPE>",
            "module": "<STIMULI_FORMAT>",
            "node_set": "<NODES_SUBSET>",
            "param1": "<VAL1>",
            "param2": "<VAL2>",
            "paramN": "value"

        }
    }

* The **<BLOCK_NAME>** is the name of the stimuli block, users can choose whatever name they want to identify a specific stimuli.
* **input_type** specifies the nature of the stimlulation, eg. if cell activity is being generated by synaptic events, current clamps, etc. The available options will depend of the resolution of the model and the type of cell models used. The options will depend on the input_type
* **module** is used to indicate the format of the stimuli. For example in trying to stimulate network with virtual spiking activity file, the individual spike times may be stored in a SONATA spikes file, a NWB file, a CSV, etc.
* **node_set** is a dictionary or reference used to select which cells to apply current stimuli to.
* Most stimuli will have one or more **parameters** options, depending on the input_type + module.


The following is a list of inputs types supported in BMTK. For further detail about how to implement a given input in 
a simulation please see the respective documentation.

.. dropdown:: Available "input_type" stimuli
  :open:

    .. list-table::
        :header-rows: 1

        * - input_type
          - module
          - description
          - available
        * - current_clamp
          - | IClamp
            | csv
            | nwb 
            | allen
          - Directly injects current into one or more cells in the network. Shape of stimulus may be a simple block, or user may specify more advanced current form using a csv, nwb, or hdf5 file.
          - BioNet, PointNet
        * - spikes
          - | sonata
            | csv
            | ecephys_probe 
            | function
          - Reads a table of spike train events into one or more virtual cells in the network.
          - BioNet, PointNet
        * - voltage_clamp
          - SEClamp
          - Applys a voltage clamping block onto one or more cells.
          - BioNet
        * - extracellular
          - | xstim
            | comsol
          - Provides an extracellular potential to alter the membrane comptanence of a selected set of cells in the network. Can replicate extracellular stimulation coming from an electrode (xstim) or field can pre-generated (comsol). 
          - BioNet
        * - replay
          - replay
          - Allows users "replay" the recurrent activity of a previous recorded simulation with a selected subset of cells and/or connections. Useful for when looking at summative properties of contributions in large networks. 
          - BioNet, PointNet
        * - syn_activity
          - | syn_activity
            | function
            | list
          - Provides spontaneous firing of a select subset of recurrently connected synapses. Users may pre-specify firing times or let bmtk generate them randomly.  
          - BioNet
        * - movie
          - movie
          - Plays a movie (eg a numpy matrix file) onto the receptive field of a grid of neurons to mimic LGN response.
          - FilterNet
        * - movie
          - | grating
            | full_field_flash
            | spontaneous
            | looming
          - Automatically generate a movie of one of a number of well-known experimental stimuli and use it to play onto the receptive field of a set of neurons.
          - FilterNet

.. card:: example network
  :class-card: .user-guide-example sd-border-2

  The "inputs" section for our 400 cell example looks like the following:

  .. code:: json
     
      "inputs": {
        "external_spikes": {
          "input_type": "spikes",
          "module": "sonata",
          "input_file": "./inputs/external_spikes.h5",
          "node_set": "external-cell"
        }
      }

  The "external_spikes" block tells BMTK that it will stimulate our network using SONATA spike-trains. The input spike 
  trains were generated by running 

  .. code:: bash 

    $ python create_inputs.py 

  which will create a SONATA spike-trains file in **inputs/external_spikes.h5**. The node-set parameter, explained 
  below, tells the module which cells we want to assign our spike-trains.



"components"
^^^^^^^^^^^^

The "components" section is used to indicate the paths to various auxiliary files required for instantiating our 
simulation; like morphology swc files, NEURON mod files, NESTML or NeuroML cell models. BMTK will use these paths to 
find neccesary files required to instantiate the network for simulation.

the different directories are defined using the format

.. code:: json 

    "components": {
        "<COMPONENT_TYPE>": "/path/to/components/dir/"
    }



.. dropdown:: Recognized "components" directories
  :open:

    .. list-table::        
        :header-rows: 1

        * - name
          - description
          - utilized by
        * - biophysical_neuron_models_dir
          - Directory containing files for instantiation of biophysically detailed models. Typically containing json 
            cell model files downloaded from the Allen Cell-Types Database.
          - **dynamics_params**
        * - point_neuron_models_dir
          - Directory containing files for instantiation of point-neuron models. Typically json parameter files, or with 
            PointNet may be model files downloaded from the Allen Cell-Types Database to instantiate optimized "glif" 
            models.
          - **dynamics_params**
        * - filters_dir
          - Directory containing parameters files for instiating models used by FilterNet.
          - **dynamics_params**
        * - morphologies_dir
          - Directory containing any morphological reconstruction files (ex. swc, neuralucdia).
          - **morphology**
        * - synpatic_models_dir
          - Directory containing files for specific synaptic parameters.
          - **dynamics_params**
        * - mechanisms_dir
          - Directory containing any morphological reconstruction files (ex. swc, neuralucdia)
          - 
        * - templates_dir
          - Contains NEURON Hoc template files or NeuroML cell or synapse models.
          - **model_template**


.. card:: example network
  :class-card: .user-guide-example sd-border-2

  For our example network, 

  .. code:: json
      
      "components": {
        "morphologies_dir": "$COMPONENT_DIR/morphologies",
        "synaptic_models_dir": "$COMPONENT_DIR/synaptic_models",
        "mechanisms_dir":"$COMPONENT_DIR/mechanisms",
        "biophysical_neuron_models_dir": "$COMPONENT_DIR/biophysical_neuron_templates/nml",
        "point_neuron_models_dir": "$COMPONENT_DIR/point_neuron_templates"
      }




"output" and "reports"
^^^^^^^^^^^^^^^^^^^^^^

The "outputs" section is where we define basic information about where and how to we will save any simulation results. Most importantly is the
**output_dir** attribute that defines default location of any files generated during the simulation. We also define the **spikes_file** attribute
which is the file name (relative to the **output_dir** path) where BMTK will save any non-virtual spikes generated during the simulation in a 
SONATA hdf5 formated file.

.. dropdown:: "output" attributes
  :open:

    .. list-table::
        :header-rows: 1

        * - name
          - description
          - default value
        * - output_dir
          - Path of output folder where simulation results and temp files will be saved in. BMTK will create the folder if it does not already
            exists. If value is not an absolute path, then will assume to be relative to location where BMTK simulation is being executed (eg `os.getcwd()`)
          - `.`
        * - overwrite_output_dir
          - If set to True then BMTK will overwrite any previous output files stored in **output_dir**. If set to False and files exists before run time then
            BMTK may throw an error and exit simulation.
          - True
        * - log_file
          - Name of file where any BMTK messages will be written to. If the file name has a relative path then file will be saved underneath **output_dir**. 
            If value is set to `false` or `none` then no log file will be created during simulation.
          - `none`
        * - log_level
          - Level of logging information that will be included during simulation (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
          - `DEBUG`
        * - log_format
          - The format string for how BMTK will save loggnig messages. It uses the `LogRecord attributes <https://docs.python.org/3/library/logging.html#logrecord-attributes>`_ options
            set by python's logging module.
          - `%(asctime)s [%(levelname)s] %(message)s`
        * - log_to_console
          - If `true` then will also log output to default **standard output (stdout)**, if `false` then will disable **stdout** logging. Note: if both **log_to_console** and
            **log_file** are set to `false` then BMTK will not log any simulation messages (simulation will still run and produce results).
          - `true`
        * - quiet_simulator
          - Can be set `true` to turn off any extermporaneous messages generated by the underlying simulator (NEURON, NEST, DiPDE)
          - `false`.
        * - spikes_file
          - location of hdf5 file where spikes will be saved. If location is a relative path file will be saved under the **output_dir** directory. If set to `none` then no
            SONATA spikes file will be created during simulation.
          - `none`
        * - spikes_file_csv
          - Location of space separated csv file where spikes will be saved. If location is a relative path file will be saved under the **output_dir** directory. If set to `none` 
            then no csv spikes file will be created during simulation.
          - `none`
        * - sort_by
          - 
          - `none`

By default BMTK will save non-virtual spike-trains of the simulation. BMTK is also capable of saving many other cell, synapse, and even network wide parameters
and attributes during a simulation, like membrane potential, Calcium concentration, or local field potentials. To instruct BMTK to record a extra parameter we 
must add one or more blocks to the "reports" subsection to config, which will have the following format:

.. code:: json 

    "reports": {
        "<BLOCK_NAME>": {
            "module": "<REPORT_TYPE>",
            "input_type": "<REPORT_VAR>",
            "cells": "<NODES_SUBSET>",
            "file_name": "<FILE_NAME>",
            "param1": "<VAL1>",
            "param2": "<VAL1>"

        }
    }

The **<BLOCK_NAME>** is a user-given identifier for a different report, each different block assumed to be independent of each other.

* **module** is used to indicate the report type and nature.
* **variable_name** indicates the specific variable in the simulation being recorded.
* **cells** is a node-set to indicate which cells are being targeted in recording.
* **file_name** is an *optional* path for where module will save output. If path is relative then it will assume to be saved under the **output_dir** 
  path specified in "output" block. If not specified, then BMTK will attempt to infer the correct path (usually under **output_dir/<BLOCK_NAME>.h5**

Different modules may also have different required/optional parameters. The following is a list of BMTK supported "report" modules:


.. dropdown:: Available "report" modules
  :open:

    .. list-table::
        :header-rows: 1

        * - module
          - description
          - available
        * - membrane_report
          - Used to record a contingous time trace of a cell ion or parameter, like membrane voltage or calcium concentration
          - BioNet, PointNet
        * - syn_report
          - Used to record a contingous time trace of variables for the synapses of a given set of cells
          - BioNet, PointNet
        * - syn_weight
          - Record of synaptic weight changes for a given set of cells. 
          - BioNet, PointNet
        * - extracellular
          - Used to record a contingous time trace of variables for the synapses of a given set of cells
          - BioNet


.. card:: example network
  :class-card: .user-guide-example sd-border-2

  For our 400 cell example we will want to have all the output generated by BMTK to be written to the **output/** folder,
  including the logging which will be written to **output/log.txt**. It will also create two spike-train files on the
  (non-virtual) cells recorded from all cells during the simulation, **output/spikes.h5** and **output/spikes.csv**. Both 
  files will contain the exact same data, only one will be stored in a SONATA hdf5 file and another in a space-separated
  csv file.

  .. code:: json

    "output":{
      "output_dir": "./output",
      "overwrite_output_dir": true,
      "log_file": "log.txt",
      "spikes_file": "spikes.h5",
      "spikes_file_csv": "spikes.csv"
    }

  Besides recording spikes, we also want to record the local field potentials of all cells plus the membrane voltage 
  traces for a select number of cells, which we do through the following two blocks in the "reports" section, 
  respectively. 

  .. code:: json
  
    "reports": {
      "vm_report": {
        "cells": "scnn1a-bio-cells",
        "variable_name": "v",
        "module": "membrane_report",
        "file_name": "vm_report.h5",
        "sections": "all"
      },

      "ecp": {
        "cells": "biophysical-cells",
        "variable_name": "v",
        "module": "extracellular",
        "electrode_positions": "./components/recXelectrodes/linear_electrode.csv",
        "file_name": "ecp.h5",
        "electrode_channels": "all",
        "contributions_dir": "ecp_contributions"
      }
    }

  The "vm_report" block will instruct BMTK to record membrane traces from all "scnn1a" type cells and save them in the 
  SONATA formated **output/vm_report.h5** file. If you wanted to record membrane potential from other cell types you 
  have the option of either modifying the "vm_report" block to save the membrane traces of other cells to the same 
  output/vm_report.h5 file. Or alternatively create another block that will independently save a different set of cell 
  voltage traces into a different file.

  The "ecp" block will record the local field potential (LFP) and save it to the file **output/ecp.h5**. 
  
  Note that recording LFP and membrane voltages at every time step can signficantly decrease simulation time and their
  resulting output files can become very large. So if you only care about spikes and firing rates then you can either
  remove these "report" blocks from the configuration file, or set attribute `"enabled: false`


"networks"
^^^^^^^^^^

The "networks" section contains path to any SONATA network files, cells and connections, used during our simulation. By 
default it is divided into two subsection, one containing any nodes (cells) files used during the simulation, and the 
other containing and edges (synapses) files used, with the following format

.. code:: json 

    "networks": {
        "nodes": [
            {
                "nodes_file": "</path/to/nodes.h5>",
                "node_types_file": "</path/to/node_types.h5>"
            }
        ],
        "edges": [
            {
                "edges_file": "</path/to/edges.h5>",
                "edge_types_file": "</path/to/edge_types.h5>"
            }
        ]
    }

BMTK will go through each nodes.h5 and edges.h5 file and import all nodes and edges population found, respectively If a 
file contains both nodes and edges populations then said file must be added to the "nodes" list and the "edges" list to 
include the total network.

.. card:: example network
  :class-card: .user-guide-example sd-border-2

  The main set of cells we want to simulate our example is saved under file name **intenral_nodes.h5**, with the 
  recurrent connections between the cells are saved in the file **internal_internal_edges.h5**. If we wanted to run 
  simulation on these and only these cells either with no input, spontaneous input, or some form of clamp we specify
  using the "inputs" section; then we could do the following in our configuration file

  .. code:: json

    "networks": {
      "nodes": [
        {
          "nodes_file": "$NETWORK_DIR/internal_nodes.h5",
          "node_types_file": "$NETWORK_DIR/internal_node_types.csv"
        }
      ],

      "edges": [
        {
          "edges_file": "$NETWORK_DIR/internal_internal_edges.h5",
          "edge_types_file": "$NETWORK_DIR/internal_internal_edge_types.csv"
        }
      ]
    }

  But in this example we explicity want to synaptically stimulate the "internal", which we can do using a separate 
  population of virtual cells we name "external". The **external_nodes.h5** contain a cell population of virtual cells 
  while the **external_internal_edges.h5** file is used to synaptically connect the "external" virtual cells to our 
  "internal" cells in a pre-determined manner.

  .. code:: json

    "networks": {
      "nodes": [
        {
          "nodes_file": "$NETWORK_DIR/internal_nodes.h5",
          "node_types_file": "$NETWORK_DIR/internal_node_types.csv"
        },
        {
          "nodes_file": "$NETWORK_DIR/external_nodes.h5",
          "node_types_file": "$NETWORK_DIR/external_node_types.csv"
        }
      ],

      "edges": [
        {
          "edges_file": "$NETWORK_DIR/internal_internal_edges.h5",
          "edge_types_file": "$NETWORK_DIR/internal_internal_edge_types.csv"
        },
        {
          "edges_file": "$NETWORK_DIR/external_internal_edges.h5",
          "edge_types_file": "$NETWORK_DIR/external_internal_edge_types.csv"
        }
      ]
    }

  When BMTK runs it will create both the "internal" and "external" population of cells and generate all external --> 
  internal and internal <--> internal synaptic connections. The "inputs" section of the configuration will assign 
  firing patterns to the "external" cells creating stimuli to our network.


"node_sets"
^^^^^^^^^^^

During a simulation we often want to apply an input or report to only a specific subset of cells. For example, we may 
want voltage traces from only pyramidal cells. We can do this can use the "node_sets" subsection to create subsets of 
our network model that can be referenced by the rest of the config

.. code:: json

  "node_sets": {
    "<SET-NAME-1>": {
      "population": "cortical",
      "node_id": [100, 101, 103, 104]
    },
    "<SET-NAME-2>": {
      "model_type": "biophysical",
      "cell_description": "pyramidal",
      "cell_location": "L23"
    }
  },

  "reports": {
    "vm_report": {
      "module": "membrane_report",
      "variable_name": "v",
      "cells": "<SET-NAME-1>",
      // ...
    },

  "inputs": {
    "iclamp_stimulus": {
          "input_type": "current_clamp",
          "module": "ICLAMP",
          "node_set": "<SET-NAME-2>",
          // ...
        }
      }


For **<SET-NAME-1>**, the node-set will tell BMTK to record from only those cells with with specified node ids. If you 
don't know the exact node_ids, or if there are too many to feasibly write down, you can filter by cell attributes. In 
**<SET-NAME-2>** we are directing BMTK to apply current clamp to all biophysical pyramidal cells found in L23. 

Users also have the option of embedding the "node-set" query parameters directly. The below example will apply inputs 
to the exact same subset of cells as done in the above.

.. code:: json

  "inputs": {
    "iclamp_stimulus": {
          "input_type": "current_clamp",
          "module": "ICLAMP",
          "node_set": {
            "model_type": "biophysical",
            "cell_description": "pyramidal",
            "cell_location": "L23"
          },
          // ...
        }
      }


.. card:: example network
  :class-card: .user-guide-example sd-border-2

  For our 400 cell example we need 3 different node-steps

  .. code:: json
    
    "node_sets": {
      "external-cells": {
        "population": "external",
        "model_type": "virtual"
      },
      "bio-cells": {
        "model_type": "biophysical"
      }
      "scnn1a-bio-cells": {
        "population": "internal",
        "model_name": "scnn1a"
      }

    }

  * The **external-cells** set will contain all cells in our "external" population of virtual cells that will
    be used in the "inputs" so that spike trains are not applied to the "internal" cells.

  * The **bio-cells** set is required by "ecp" recording block. Our model contains both morpholigcally detailed
    and point-neuron models, but only the former can be used to record the local field potential. Passing "node_set": 
    "bio-cells" to the ecp module will make sure the module doesn't try (and crash) trying to record from cell models
    that don't produce extracellular potential.

  * The **scnn1a-bio-cells** set is used by the "vm_report" recording block so that it knows only to record voltage 
    traces that have their "model_name" attribute set to value "scnn1a". Although we could record "v" varaible from 
    all cells, doing so would increase simulation time and generate a lot of extra data we don't need. 


"manifest" **[OPTIONAL]**
^^^^^^^^^^^^^^^^^^^^^^^^^

The "manifest" section lets users define variables and special directives to be used throughout the rest of the 
configuration file. SONATA uses the standard "$" posix prefix for differentiating a constant versus a varaible.

For example, we use the manifest in the following manner to create custom variable "$NETWORK_DIR"

.. code:: json

   "manifest": {
      "$NETWORK_DIR": "$/path/to/my/models/network/"
   },
   
   "networks": {
      "nodes": [
        {
          "nodes_file": "$NETWORK_DIR/network1_nodes.h5",
          "node_types_file": "$NETWORK_DIR/network1_types.csv"
        },
        {
          "nodes_file": "$NETWORK_DIR/network2_nodes.h5",
          "node_types_file": "$NETWORK_DIR/network2_types.csv"
        }
      ]
    }

This way if we need to change the location of our network files or copy it to a new drive we can simply update the 
manifest in one single place. 


Splitting the config **[OPTIONAL]**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is convient to think of the SONATA configuration as two parts; the *simulation* section with the "run", 
"inputs", and "reports" section, and the *network* section with the "networks" and "components" sections. BMTK allows
configuration file to be split up along these lines, first by splitting them up into respective json files, for 
example called *./path/to/config.simulation.json* and *./path/to/config.network.json* (you can use whatever name and 
path you choose). Then to import these two sections into one file just use the following:

.. code:: json 

  {
    "simulation": "./path/to/config.simulation.json",
    "network": "./path/to/config.network.json"
  }

And when BMTK runs it will locate and combine both configuration into one json dictionary. 

If you want to reduce the number of files you can also import a separate "network" config into a "simulation" config 
(or vice-versa). 

.. code:: json

  {
    "run": {
      // ...
    },
    "inputs": {
      // ...
    },
    "reports": {
      //...
    },

    "network": "./path/to/config.network.json"
  }  


This can be useful if you have multiple simulations on the same network with different input regimens. It insures that
each simulation configs all use the same model. And if you have to update the model and/or component paths then you 
only need to do so once. 



3. Run the Simulation
=====================

Now that we have all the files we need to instantiate the model, along with a configuration that specifies the run-time,
inputs, and reporting conditions of our simulation; we can go ahead an execute the simulation using the following 
command:


.. code:: bash

  $ python run_simulation.py config.simulation.json


If you have a machine or cluster with multiple cores BMTK will automatically parallize the simulation. The only change 
the user will have to make is to change the way python is called to comply with your setup.

.. code:: bash

  $ mpirun -np N python run_simulation.py config.simulation.json


The simulation can also be executed inside Jupyter notebook/lab if prefered. Just copy the contents of the run_script 
(see below) into the cell and execute as you would a normal notebook. 

There are numerious other options for ways to execute BMTK; including using:

* **Docker Image**: The BMTK Docker image allows users to run BMTK through a virtual machine containing a pre-generated
  environment will BMTK and all dependencies installed. You can use `docker run` command to run a simulation on your 
  machine, or even spin up a Jupyter lab server (that will include BMTK tutorials and examples).

  See `Using Docker with BMTK` guide for more information.

* **Neuroscience Gateway (NSG)**

* **singularity**

run script
^^^^^^^^^^

The for majority of use-cases users will not need to do any programming, as most simulation options can be set using 
the SONATA simulation configuration file. But BMTK does makes available for users to insert customized code and data 
into any simulation. To do so you will need to interface with and add to the default run_simulation script used to 
execute and simulate a network.

If you want to modify the way BMTK executes, and/or run a BMTK simulation from a Jupyter cell or within another 
piece of code, use the code within the **run()** function . **note**: there is slight differences in the run script 
depending on the underlying "simulation engine" used, although they all follow the same structure.

.. tab-set::

  .. tab-item:: BioNet

    .. code:: python
      :linenos:
      
      import sys
      from bmtk.simulator import bionet

      def run(config_path):
          config = bionet.Config.from_json(config_path)
          config.build_env()

          graph = bionet.BioNetwork.from_config(config)
          sim = bionet.BioSimulator.from_config(config, network=graph)
          sim.run()
          
          bionet.nrn.quit_execution()

      if __name__ == '__main__':
          run(sys.argv[-1])

    **description of run() function by lines #** 
    
    5. Loads in the **config.simulation.json** file into python a dictionary-like object, resolving any variables, 
       paths, and directives.  
    
    6. Sets up logging and creates and validates any directories/files to be used for recording simulation output.
    
    8. Initializes the network using the SONATA configuration file's "networks" section.
    
    9. Initialize the simulation.
    
    10. Runs the simulation. 

    11. Because BioNet runs within a NEURON console, we must explicity exit() it.


  .. tab-item:: PointNet 

    .. code:: python 
      :linenos:

      import sys
      from bmtk.simulator import pointnet

      def run(config_path):
          config = pointnet.Config.from_json(config_path)
          config.build_env()

          network = pointnet.PointNetwork.from_config(config)
          sim = pointnet.PointSimulator.from_config(config, network)
          sim.run()

      if __name__ == '__main__':
          run(sys.argv[-1])

    **description of run() function by lines #** 
    
    5. Loads in the **config.simulation.json** file into python a dictionary-like object, resolving any variables, 
       paths, and directives.  
    
    6. Sets up logging and creates and validates any directories/files to be used for recording simulation output.
    
    8. Initializes the network using the SONATA configuration file's "networks" section.
    
    9. Initialize the simulation.
    
    10. Runs the simulation. 


  .. tab-item:: FilterNet

    .. code:: python
      :linenos:

      import sys
      from bmtk.simulator import filternet

      def run(config_path):
          config = filternet.Config.from_json(config_path)
          config.build_env()

          net = filternet.FilterNetwork.from_config(config)
          sim = filternet.FilterSimulator.from_config(config, net)
          sim.run()

      if __name__ == '__main__':
          run(sys.argv[-1])

    **description of run() function by lines #** 
    
    5. Loads in the **config.simulation.json** file into python a dictionary-like object, resolving any variables, 
       paths, and directives.  
    
    6. Sets up logging and creates and validates any directories/files to be used for recording simulation output.
    
    8. Initializes the network using the SONATA configuration file's "networks" section.
    
    9. Initialize the simulation.
    
    10. Runs the simulation. 


  .. tab-item:: PopNet

    .. code:: python
      :linenos:

      import sys
      from bmtk.simulator import popnet

      def run(config_path):
          configure = popnet.config.from_json(config_path)
          configure.build_env()

          network = popnet.PopNetwork.from_config(configure)
          sim = popnet.PopSimulator.from_config(configure, network)
          sim.run()

      if __name__ == '__main__':
          run(sys.argv[-1])


    **description of run() function by lines #** 
    
    5. Loads in the **config.simulation.json** file into python a dictionary-like object, resolving any variables, 
       paths, and directives.  
    
    6. Sets up logging and creates and validates any directories/files to be used for recording simulation output.
    
    8. Initializes the network using the SONATA configuration file's "networks" section.
    
    9. Initialize the simulation.
    
    10. Runs the simulation. 


simulation results
^^^^^^^^^^^^^^^^^^

Depending on the complexity of the model and inputs and reports, a simulation may take anywhere between a few seconds 
to a few days to complete. By default, BMTK will automically save the any results and "reports" as set up in the 
SONATA config. Once completed and results are saved we can go ahead and analyze our results, which we go into Further
details in the `NEST <Analysis>`_ section of the user guide.


.. card:: example network
  :class-card: .user-guide-example sd-border-2

  We can go ahead and simply run our 400 cell network simulation using the **run_simulation.py** script found in our 
  working folder. 

  .. tab-set::

    .. tab-item:: single core

      The simplest way to run the simulation in a command line using a single core.

      .. code:: bash 

        $ python run_simulation.py config.simulation.json

    .. tab-item:: multiple-cores using MPI

      If you have MPI (Message Passing interface) installed on you machine use the following to split the simulation up
      between *N* cores (replace *N* with the number of cores/ranks).

      .. code:: bash 

        $ mpirun -np N nrniv -mpi -python run_simulation.py config.simulation.json


    .. tab-item:: In a jupyter lab cell

      To run the simulation inside jupyter notebook, add the following lines to a cell and execute:
      
      .. code:: python 

          from bmtk.simulator import bionet
          config = bionet.Config.from_json("config.simulation.json")
          config.build_env()
          graph = bionet.BioNetwork.from_config(config)
          sim = bionet.BioSimulator.from_config(config, network=graph)
          sim.run()

    .. tab-item:: Docker

      If you have docker client installed on your machine, you can use the following to execute the simulation

      .. code:: bash 

        $ docker run alleninstitute/bmtk -v local/path:/home/shared/workspace python run_simulation.py config.simulation.json



  When it starts the first thing it will do is create (or overwrite) the **output/** folder along with 
  **output/log.txt** log file that we can use to keep track of progress of the simulation. Although the network is 
  small enough to run on any modern computer or laptop, it will still take anywhere 5 to 30 minutes to complete 
  depending on the hardware.

  Once completed it will create the following files:

  * **output/spikes.h5** and **output/spikes.csv** contain spike trains for all non-virtual cells. (Both files have the
    same data but in different file formats).

  * **output/vm_reports.h5** contains membrane traces of selected cells in SONATA formated hdf5 file.

  * **ouput/ecp.h5** contains local field potential (LFP) recordings from all the biophysically detailed cells, as 
    recorded from a simulated electroded, and saved in SONATA hdf5 format.

   

Simulation Engines
==================

As mentioned before, BMTK is capable of simulating a wide variety of different network models across multiple levels 
of cell resolutions. To do this BMTK utilizes different backend simulator libraries, or "Simulation Engines", depending
on the nature of the model (ie. compartmental models, point models, filter models, rates based models, etc).

BMTK standarizes and abstracts the simulation process so that users can easily switch between models types with having 
to learn a whole new API, however there are still important differences between the varying Simulation Engines. Some 
may excpect certain parameters and attributes (e.g. compartmental models will expect cells to have a defined morphology).
Similarily some may contain extra features and capabilities.

To learn more about the requirements and capabilities for the model(s) of your attention please see their respective 
user guides.


.. grid:: 1 1 4 4
    :gutter: 1

    .. grid-item-card::  
        :link: bionet.html
        :img-bottom: _static/images/bionet_rep_morpholgy_network.png

        **BioNet** - Multicompartment Biophysicaly Detailed Simulation
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        *BioNet* utilizes the NEURON simulator tool to allow simulation of multicompartment cell models. It can 
        encorporate a cells full morphology into model and the simulation; allowing you to simulate apects including ion 
        flow, intracellular and extracellular membrane comptanence, and synaptic location and density. 

    .. grid-item-card::
        :link: pointnet.html
        :img-bottom: _static/images/pointnet_figure.png

        **PointNet** - point-neuron based models
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        *PointNet* utlizes the NEST simulator for simulation of networks of point-neuron models, including Allen's 
        glif models, izhieckcih models, hodgkin-huxley, and many more. Most PointNet network models run faster with 
        less overhead than BioNet and is a good starting point.

    .. grid-item-card::
        :link: filternet.html
        :img-bottom: _static/images/filternet_rep_filter_models.png

        **FilterNet** - Receptive Field Filter Models
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        *FilterNet* allows modelers to play visual or auditory onto a receptive field model to generate realistic 
        firing rates and spike trains based on the spatio-temporal properties of the stimuli. The results of which can 
        be used by compartmental and point models as realistic stimuli.


    .. grid-item-card::
        :link: popnet.html
        :img-bottom: _static/images/dipde_figure.png

        **PopNet** - Population Wide Firing Rates Models
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        *PopNet* - Use's Allen's DiPDE solver to look at population level firing rate dyanmics.



Additional Resources and Guides
===============================

Exectuion and Run Options
^^^^^^^^^^^^^^^^^^^^^^^^^

The following guides and tutorials for setting up and running simulations across a wide variety of different 
computing environments. 

.. grid:: 1 1 4 5
    :gutter: 1

    .. grid-item-card:: Running BMTK on the on HPC.
        :link: builder_guide 

        Guides, tips and tricks for running bmtk using high-performance computing resources with Message Passing 
        Interface (MPI). Including tips for installing MPI, running on slurm, and using singularity container.

    .. grid-item-card:: Running BMTK on the Neuroscience Gateway (NSG) 
        :link: builder_guide 

        The Neuroscience Gateway is a tool for neuroscientists to access HPC resources for free. Guide shows how to 
        use BMTK with the NSG web and API interface to build networks and run large simulations.

    .. grid-item-card:: Cybershuttle
        :link: builder_guide 

        Comming Soon.

    .. grid-item-card:: Using BMTK Images and Applications
        :link: builder_guide 

        How to setup and run BMTK using pre-built applications and images; including with Docker, AppImage, and Snap. 
   
    .. grid-item-card:: Parallel and Serial Simulations 
        :link: builder_guide 

        How to run multiple simulations in parallel and/or serial; including grid searching, evolutionary, and gradient
        search methods for optimizing network and simulations. 


Inputs
^^^^^^

BMTK supports using a wide variety of inputs and stimulus when running a simulation. Please see the following guides 
for built-in "inputs" types and how to use them in your simulations.


Reports
^^^^^^^

Modelers can choose which variables and changes in a simulation to record using the "reports" section. See following 
guides for further information how to implement such output.


Useful Options and Scripts
^^^^^^^^^^^^^^^^^^^^^^^^^^


Advanced Features
^^^^^^^^^^^^^^^^^
