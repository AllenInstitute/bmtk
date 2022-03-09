# 14 Cell network

A small network of 14 cells - 10 multi-compartment biophysically detailed cells and 4 point integrate-and-fire cells -
called V1. Recieves input from two networks of virtual cells (spike-trains) - LGN and TW. Uses the BioNet simulator 
(requires NEURON)

This simulation is an example of one way to use bmtk to do automated parameter optimization, both by 
itself and in conjunction with other software. In particular software like [nested](https://github.com/neurosutras/nested).
The simulation is ran multiple times in a row, each time the firing rate averages of each "pop_name" type (eg Scnn1a,
PV1, etc) are calculated and compared to their target firing rates (```target_frs``` variable in run_bionet.py script).
The synpatic weights are then updated using a primative gradient method. 

After the N simulations have been completed, the MSE of the actual vs target firing-rates are calculated and the updated
weights are saved to **updated_weights/** directory.

## Running:
To run a simulation, install bmtk and run the following:
```
$ python run_bionet.py config.simulation.json
```
If successful, will create a *output* directory containing log, V1 spike trains and recorded cell variables.

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




