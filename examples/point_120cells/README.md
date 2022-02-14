# 120 point neuron network

A small network of 120 point-neurons. Uses PointNet and will require NEST to run.


## Running:
To run a simulation, install bmtk and run the following:
```
$ python run_pointnet.py config.simulation.json
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

## Perturbation simulations
The file ```config_pertubration.json``` uses the same network (recurrent connections + feedforward inputs) as before. However
is designed to also simulate the effects of optogenetic or current clamp inhibition and excitation on a subset of the cells. 
To run this example:
```bash
$ python run_pointnet.py config_perturbations.json
``` 

The only difference between this simulation and the original is in the **inputs** section in the config. To simulate the 
perturbations we use a current clamp input to make cells 20 through 39 overly excitatory:
```json
"exc_perturbation": {
   "input_type": "current_clamp",
   "module": "IClamp",
   "node_set": {
     "population": "cortex",
     "node_ids": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
   },
   "amp": 230.0,
   "delay": 1.0,
   "duration": 3000.0
},
```

For cells 40 through 49 we can use a negative current to depress the cell activity
```json
"inh_perturbation": {
  "input_type": "current_clamp",
  "module": "IClamp",
  "node_set": {
    "population": "cortex",
    "node_ids": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
  },
  "amp": -230.0,
  "delay": 1.0,
  "duration": 3000.0
}
```
