# The EI-network

Network of 12,500 randomly connected excitatory and inhibitory neurons. This example provides the same network with 
comparable dynamics using multi-compartmental models (BioNet), glif point-neuron models (PointNet), and population 
models (PopNet).

## Running the models

The scripts for building the network and running the simulations can be found in each simulators subdirectory. To
build and run a simulation for BioNet:

```bash
	$ cd bionet
	$ python build_network.py
	$ python run_bionet.py config.json
```

And for PointNet (or PopNet) replace "bionet" with "pointnet" (or "popnet").


## Analyzing the results

Unless changed in the *config.json* file, simulation results will be saved to the *output* directory by default. BioNet
and PointNet will produce spike-trains for each neuron. While PopNet will produce the firing-rate dynamics of the 
two populations.


