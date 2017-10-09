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
    "tstop": 3000.0,					# run time in (ms)
    "dt": 0.1,							# time step in (ms)	
    "dL": 20,							# length of compartments (i.e., segments) in (um)
    "overwrite_output_dir": true,		# if True: will overwrite the output directory; if False: will issue an error that directory exists
    "spike_threshold": -15,				# block
    "nsteps_block":5000,
    "save_cell_vars": ["v", "cai"],
    "calc_ecp": true
  },
```
