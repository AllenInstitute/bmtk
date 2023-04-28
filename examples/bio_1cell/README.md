# 1 Cell network

A single excitatory cell that is stimulated by current clamps or virtual synapses

Uses the BioNet simulator (requires NEURON)

## Compiling NEURON mechanisms
The components for the BioNet examples are located in /examples/bio_components. If the NEURON mechanisms have not already been compiled, the following should compile the NEURON mechanisms and place them in another folder in /mechanisms.

```bash
$ cd ../bio_components/mechanisms
$ nrnivmodl modfiles 
$ cd -
```
Failure to compile the mechanisms results in an error such as:
```
**Warning**:  NEURON mechanisms not found in ./../bio_components/mechanisms.
              [...]
              ValueError: argument not a density mechanism name
```


## Running:
To run a simulation using a current clamp, install bmtk and run the following:
```bash
$ python run_bionet.py config.simulation_iclamp.json
```
If successful, will create a *output_iclamp* directory containing log, V1 spike trains and recorded cell variables.

You can also use the "virt" network which will use a virtual cell to stimulate the biophysical cell with synaptic
stimulation.

```bash
$ python run_bionet.py config.simulation_syns.json
```

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


