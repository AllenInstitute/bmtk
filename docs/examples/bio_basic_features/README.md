# BioNet input simulation

A simple non-connected network of 5 neuron (visual cortex mouse models taken from the Allen Cell-Types 
database) network with various inputs including current clamp, external synaptic inputs, and extracellular
electrode stimulation.

## Building the network.
Network files are stored in network/bio_nodes.h5 and network/bio_node_types.csv. In the same directory you'll
also find network files representing virtual input (virt*). You can rebuild the network by running
```bash
$ python build_network.py
```

## Simulations
### Current clamp
```bash
$ python run_bionet.py config_iclamp.json
```

Will run a series of current injections into each cell. Current clamp parameters are set in config_iclamp.json
(under the "inputs" section). Output files will by default be written to output_iclamp/.

### Spike stimulations
```bash
$ python run_bionet.py config_spikes_input.json
```

Runs a simulation where each biophysical cells recieve spike trains from external cells. It uses a separate network
(network/virt_node*) with synaptic weight/location set in the files network/virt_bio_edges.h5 and 
network/virt_bio_edge_types.csv, with spike trains set in inputs/exc_spike_trains.h5

### Extracellular stimulation
```bash
$ python run_bionet.py config_xstim.json
```

Runs a simualtion where all the cells are stimulated by a extracellular electrode. Extracell stimulation parameters
are set in config_xstim.json
