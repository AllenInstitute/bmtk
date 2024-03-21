# bio_all_active_sweep

Contains an example of how to download and "Biophysical, all-active" models from the 
(Allen Cell Types Database)[https://celltypes.brain-map.org/data]. Also includes examples of how to 
alter the "model_processing" function so that you can run cells with their full axon rather than with
the replacement stubs that is created by default (warning: model parameters were optimized by replacing
morphologicial axon with stubs)

## Directory Structure
* ephys_inputs/ - Folder containing NWB files downloaded for the Allen Cell Types Database. These files contain different experimental stimulus sweeps (eg square-waves, ramps, noise). Used during simulation to create current clamp stimulus onto the cells.
* network/ - Folder containing SONATA network files, created by the `build_network.py` script and used when running the 
* models/ - Folder containing model files (especially parameters json file and morphology swc) downloaded from Allen Cell Types Database.
* output\*/ - Folders containing results from simulations, created by running the run_bionet.py scripts. By default contains simulation spike-times and soma membrane voltage potentials.
* build_network.py - Python script to (re)build the SONATA network/ files, can also be used to download models/ and ephys_inputs/ files.
* run_bionet.py - Python script to execute a simulation.
* config.simulation.\*.json - SONATA config file containg all the information bmtk will require to run a full simulation - including location of network files, download Cell Types DB files, simulation parameters, etc.



## (Re)Building the network

To download the Cell-Type files and create the SONATA network files required to run a simulation sweep you can run:
```bash
    $ python build_network.py
```

If you want to try running a simulation with a different cell model just replace the `specimen_id` value in the python file. The script will also attempt to automatically download the model files and ephys_data nwb files if not already in the folder (if you've manually download these files the script will not try to download it again).

You can also change `axon_type` to either **full** or **stub**. The default behavior is a stub axon (eg. It removes the axon and replaces it with a simple small column). But you can change it full in which case when the simulation runs it will simulate the cell using full morphology reconstruction.



## Running the simulation

To run a simulation with a given `config.simulation.\*.json` file run the following python command:

```bash
    $ python run_bionet.py config.simulation.json
```

BMTK will automatically load in the configuration file, run a full simulation, and save the results into the output\*/ folder (as specified in the configuration json).

If you want to run a simulation with a different cell model, sweep number, or axon type you can do the following:
1. Copy one of the existing `config.simulation.\*.json` files and open it up with your preferred text editor.
2. In the "manifest" section update the following params as needed.
   1. **SWEEP_NUM** - Change to run with a different stimulus sweep.
   2. **AXON_TYPE** - Set to either *fullaxon* or *stubaxon* depending on how to handle the model's axon.
   3. **MODEL_ID** - Set the the model being used.
3. For some stimulus sweeps you may also need to adjust **run/tstop** (the run time in ms) otherwise the simulation may stop before sweep has finished.

