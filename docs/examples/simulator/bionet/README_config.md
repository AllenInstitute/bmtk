# Structure of configuration file
The configuration file is broken into several groupings:

The "manifest" grouping includes path variables that will be recursively built during the execution to provide the absolute paths. These paths then can be used to to specify the directories and file names in the remainder of the configuration file 
```
  "manifest": {
    "$BASE_DIR": "${configdir}",
    "$OUTPUT_DIR": "$BASE_DIR/output",
    "$INPUT_DIR": "$BASE_DIR/../../NWB_files",
    "$NETWORK_DIR": "$BASE_DIR/network",
    "$COMPONENT_DIR": "$BASE_DIR/../components"
  },
```
Here variable "${configdir}" stands for the directory where configuration file is located on the file system. User may specify any other valid directory path to serve as a base directory $BASE_DIR.

The "run" grouping includes run-time parameters with the corresponding brief explanation


```
  "run": {
    "tstop": 3000.0,					# run time (ms)
    "dt": 0.1,							# time step (ms)	
    "dL": 20,							# length of compartments (i.e., segments) (um)
    "overwrite_output_dir": true,		# if True: will overwrite the output directory; if False: will issue an error that directory exists
    "spike_threshold": -15,				# cells exceed this value of spike threshols (mV) will issue a spike
    "nsteps_block":5000,				# will save output variables in memory for the nsteps_block and then write to disk
    "save_cell_vars": ["v", "cai"],		# save somatic variables in the list
    "calc_ecp": true					# calculate ExtraCellular Potential (ECP): True or False
  },
```

The "conditions" grouping includes information about the initial conditions:

  "conditions": {
    "celsius": 34.0,					# temperature (C)
    "v_init": -80						# initial membrane voltage (mV)	
  },


The "node_id_selections" defines group selections. For example, this way we identify the cells for which variables will be saved
  "node_id_selections": {
    "save_cell_vars": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
  },

The "input" gropuing defines the spike_trains for the the external inputs

```
  "input": [
    {
      "type": "external_spikes",
      "format": "nwb",
      "file": "$INPUT_DIR/lgn_spikes.nwb",
      "network": "LGN",
      "trial": "trial_0"
    },
	...
  ],
```

The "output" grouping defines file names for the output

```
  "output": {
    "log_file": "$OUTPUT_DIR/log.txt",				#	log file
    "spikes_ascii_file": "$OUTPUT_DIR/spikes.txt",	#	file for spikes in ascii format
    "spikes_hdf5_file": "$OUTPUT_DIR/spikes.h5",	#	file for spikes in HDF5 format
    "cell_vars_dir": "$OUTPUT_DIR/cellvars",		#	folder to save variables from individual cells
    "ecp_file": "$OUTPUT_DIR/ecp.h5",				#	file to save ExtraCellular Potential
    "output_dir": "$OUTPUT_DIR"						#	output directory
  },
```

```
  "target_simulator":"NEURON",
```

The "components" grouping includes paths to directories containing building blocks of a network model
```
  "components": {
    "morphologies_dir": "$COMPONENT_DIR/biophysical/morphology",	# morphologies	
    "synaptic_models_dir": "$COMPONENT_DIR/synaptic_models",		# synaptic models
    "mechanisms_dir":"$COMPONENT_DIR/mechanisms",					# NEURON mechanisms
    "biophysical_neuron_models_dir": "$COMPONENT_DIR/biophysical/electrophysiology", # parameters of biophysical models
    "point_neuron_models_dir": "$COMPONENT_DIR/intfire",			# parameters of point neuron models	
    "templates_dir": "$COMPONENT_DIR/hoc_templates"					# NEURON HOC templates
  },
```

The "recXelectrode" grouping includes parameters defining extracellular electrodes
```
  "recXelectrode": {
	"positions": 		"$COMPONENT_DIR/recXelectrodes/linear_electrode.csv"	# file with positions of recording electrodes
	},
```

The "networks" grouping includes files defining nodes and edges:

```
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
```
