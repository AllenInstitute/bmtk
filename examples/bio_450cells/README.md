# 450 cells network

This is a small example network of a 450 cell simulation based on the 45,000 mouse layer 4 network described in
Arkhipov et. al. 2018. Of the cells 180 are biophysically detailed multicompartment cells downloaded from the
Allen Cell-Types database, the remaining are point integrate-and-fire neuron models. The network is driven by
an external network of virtual nodes/spike-trains.

Uses BioNet simulator and will require NEURON to run.

This example is different from the other bio_450cell example in that:
 * The network synaptic locations are explicity defined.
 * Includes LFP recordings.  NOTE: this will significantly increase the time to complete the simulation and will
 write a large ecp results file in the output.


## Running:
To run a simulation, install bmtk and run the following:
```bash
$ python run_bionet.py config.simulation.json
```

to run using multiple cores using MPI:
```bash
$ mpirun -np <N> nrniv -mpi -python run_bionet.py config.simulation.json
```

If successful, will create a *output* directory containing log, spike trains and recorded cell variables. By default
only the "internal" spike times are recorded to a spikes.h5 file - as specified in the config. The config ```config.simulation_ecp.json``` 
will also record extra-cellular potentials, and ```config.simulation_vm.json``` will record membrane potential variable
from the soma.

## The Network:
The network files have already been built and stored as SONATA files in the *network/* directory. The bmtk Builder
script used to create the network files is *build_network.py*. To adjust the parameters and/or topology of the network
change this file and run:
```
$ python build_network.py
```
or
```bash
$ mpirun -np <N> python build_network.py
```

This will overwrite the existing files in the network directory. Note that there is some randomness in how the network
is built, so expect (slightly) different simulation results everytime the network is rebuilt.

## Simulation Parameters
Parameters to run the simulation, including run-time, inputs, recorded variables, and networks are stored in config.json
and can modified with a text editor.

## Plotting results
```
$ python plot_output.py
```