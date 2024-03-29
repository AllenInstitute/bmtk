# Excitatory/Inhibitory Network for BioNet

## Requirements

* BMTK 
* Neuron 7.4+

## Files
* network/ - Circuit file of E/I cells and external network. If folder doesn't exists use the build_network.py script.
* inputs/ - Spike-train files corresponding to the external network used to drive the simulation.
* config.json - simulation and network parameters.
* build_network.py - Used to build the network/ files.
* build_input.py - Used to build the inputs/ file(s).
* run_bionet.py - Script for running the network simulation.


## Building the Network

The Network model files must be built for running the simulation using the build_network.py script. It may take 
anywhere between 5-10 minutes to complete and will save all required files in network/ folder.

```bash
	$ python build_network.py
```

## Running the Simulation

You can use config.json to change parameters for simulating the network. By default everything recorded during the
simulation is saved in the output/ directory, including a running log.txt to keep track of progress. 

```bash
	$ python run_bionet.py config.json
```

The network may have problems running on a single computer core. It is recommended to use a computing  cluster, or 
if running on a single machnie use MPI to run on multtple cores:
```bash
	$ mpirun -n <N> nrniv -mpi -python run_bionet.py config.json
```