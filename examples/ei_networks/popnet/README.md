# Excitatory/Inhibitory Network for PopNet

## Requirements

* BMTK 
* NEST 2.12+

## Files
* network/ - Circuit file of E/I cells and external network. If folder doesn't exists use the build_network.py script.
* inputs/ - Spike-train files corresponding to the external network used to drive the simulation.
* config.json - simulation and network parameters.
* build_network.py - Used to build the network/ files.
* run_popnet.py - Script for running the network simulation.


## Building the Network

The Network model files must be built for running the simulation using the build_network.py script.

```bash
	$ python build_network.py
```

## Running the Simulation

You can use config.json to change parameters for simulating the network. By default everything recorded during the
simulation is saved in the output/ directory, including a running log.txt to keep track of progress. 

```bash
	$ python run_popnet.py config.json
```