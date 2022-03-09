# 450 cells network

This is a small example network of a 450 cell simulation based on the 45,000 mouse layer 4 network described in
Arkhipov et. al. 2018. Of the cells 180 are biophysically detailed multicompartment cells downloaded from the
Allen Cell-Types database, the remaining are point integrate-and-fire neuron models. The network is driven by
an external network of virtual nodes/spike-trains.

Uses BioNet simulator and will require NEURON to run.


## Running:
To run a simulation, install bmtk and run the following:
```
$ python run_bionet.py config.json
```
If successful, will create a *output* directory containing log, spike trains and recorded cell variables.

## The Network:
The network files have already been built and stored as SONATA files in the *network/* directory. The bmtk Builder
script used to create the network files is *build_network.py*. To adjust the parameters and/or topology of the network
change this file and run:
```
$ python build_network.py
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