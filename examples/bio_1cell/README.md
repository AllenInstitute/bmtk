# 1 Cell Biophysical Network Model

An Example of a biophysical network containing a single biophysical morphologically detailed cell. The cell recieves stimuli either from a current clamp at the soma (using *config.simulation_iclamp.json*) or with synaptic stimuli using virtual cells with predefined spike-trains (*config.simulation_syns.json*).


## Requirements
- BMTK
- NEURON 8.0+


## Important Files

- **inputs/** - folder contains pre-generated spike-train files use as stimuli for *config.simulation_iclamp.json*
- **network/** - Folder containing network model.
- **config.simulation_iclamp.json** - Configuration file with instructions for running network driven by current clamp stimuli attached at the soma.
- **config.simulation_syns.json** - Configuration file with instructions for running network driven by virtual cells that synaptical stimulate our network.
- **build_network.py** - script to rebuild the "network". WARNING: By default will override the **network/** folder.
- **run_bionet.py** - Main script for running BioNet simulation of the network.


## Running the Simulation

[See Instructions for running BioNet Simulation](../README.md#bionet)

