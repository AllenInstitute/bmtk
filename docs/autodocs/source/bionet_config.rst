:orphan:

Configuration file
==================

The configuration file defines the files describing the network, its input as well as run-time parameters. For convenience it is grouped into the following categories:


Manifest
++++++++

    Includes custom path variables that may be used to build full paths to files. The special variable "${configdir}" in the "$BASE_DIR" stands for the directory where the configuration file is located. Users may specify any valid directory for "$BASE_DIR" as well.

    
    ::
    
          "manifest": {
            "$BASE_DIR":    "${configdir}",
            "$OUTPUT_DIR":  "$BASE_DIR/output",
            "$INPUT_DIR":   "$BASE_DIR/../../NWB_files",
            "$NETWORK_DIR": "$BASE_DIR/network",
            "$COMPONENT_DIR":"$BASE_DIR/../components"
          },
   

Run
+++

    Includes run-time parameters 
    
    ::

          "run": {
            "tstop":3000.0,         # run time (ms)
            "dt": 0.1,          # time step (ms)    
            "dL": 20,           # length of compartments (i.e., segments) (um)
            "overwrite_output_dir": true,   # if True: will overwrite the output directory; if False: will issue an error that directory exists
            "spike_threshold": -15,     # will record a spike when membrane voltage (mV) is exceeded
            "nsteps_block":5000,        # will write to disk data after this many steps
            "save_cell_vars": ["v", "cai"], # save somatic variables in the list
            "calc_ecp": true        # calculate ExtraCellular Potential (ECP): True or False
          },


Conditions
++++++++++

    Includes information about the initial conditions:
    
    ::


          "conditions": {
            "celsius": 34.0,    # temperature (C)
            "v_init": -80       # initial membrane voltage (mV) 
          },


Node_id selections
++++++++++++++++++

    Defines selections of cells. For example, this way can specify the cells (node_ids) for which variables will be saved
    
    ::

          "node_id_selections": {
            "save_cell_vars": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
          },


Input
+++++

    Defines spike_trains for the external inputs


    ::

          "input": [
            {
              "type": "external_spikes",
              "format": "nwb",
              "file": "$INPUT_DIR/lgn_spikes.nwb",
              "source_nodes": "LGN",
              "trial": "trial_0"
            },
            ...
            ]

Output
++++++

    Defines file names for the output


    ::

      "output": {
        "log_file":     "$OUTPUT_DIR/log.txt",      # log file
        "spikes_ascii_file":    "$OUTPUT_DIR/spikes.txt",   # file for spikes in ascii format
        "spikes_hdf5_file": "$OUTPUT_DIR/spikes.h5",    # file for spikes in HDF5 format
        "cell_vars_dir":    "$OUTPUT_DIR/cellvars",     # folder to save variables from individual cells
        "ecp_file":     "$OUTPUT_DIR/ecp.h5",       # file to save extracellular potential
        "output_dir":       "$OUTPUT_DIR"           # output directory
      },




Components
++++++++++

    The "components" grouping includes paths to directories containing building blocks of a network model

    ::

          "components": {
            "morphologies_dir":     "$COMPONENT_DIR/biophysical/morphology",    # morphologies  
            "synaptic_models_dir":      "$COMPONENT_DIR/synaptic_models",       # synaptic models
            "mechanisms_dir":       "$COMPONENT_DIR/mechanisms",            # NEURON mechanisms
            "biophysical_neuron_models_dir":"$COMPONENT_DIR/biophysical/electrophysiology", # parameters of biophysical models
            "point_neuron_models_dir":  "$COMPONENT_DIR/intfire",           # parameters of point neuron models 
            "templates_dir":        "$COMPONENT_DIR/hoc_templates"          # NEURON HOC templates
          },


Recording Extracellular Electrode
+++++++++++++++++++++++++++++++++

    Includes parameters defining extracellular electrodes


    ::

          "recXelectrode": {
            "positions": "$COMPONENT_DIR/recXelectrodes/linear_electrode.csv"   
            },




Networks
++++++++

    Includes files defining nodes and edges:

::

      "networks": {
        "nodes": [
          {
            "name": "V1",                                                                                       
            "nodes_file": "$NETWORK_DIR/v1_nodes.h5",
            "node_types_file": "$NETWORK_DIR/v1_node_types.csv"
          },
            ...
        ],

        "edges": [
          {
            "target": "V1",
            "source": "V1",
            "edges_file": "$NETWORK_DIR/v1_v1_edges.h5",
            "edge_types_file": "$NETWORK_DIR/v1_v1_edge_types.csv"
          },
        ...
        ]
      }


