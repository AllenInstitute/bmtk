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
Stimulates the network using a series of current clamps at all the biophysically detailed cell models  

```json
  "inputs": {
    "current_clamp_1": {
      "input_type": "current_clamp",
      "module": "IClamp",
      "node_set": "biophys_cells",
      "amp": 0.1500,
      "delay": 500.0,
      "duration": 500.0
    }
```
Times are in miliseconds and the delay is from the begining of the simulation. To run:


```bash
$ python run_bionet.py config.simulation_iclamp.json
```

Will run a series of current injections into each cell. Current clamp parameters are set in config_iclamp.json
(under the "inputs" section). Output files will by default be written to output_iclamp/.

### Spike stimulations
Stimulates the network using "virtual" cells (**virt** network) that synapse onto the biophysically detailed cell models 
and activate using pre-set spike-trains. The spike trains for the **virt** network cells are stored in 
*inputs/virt_spikes.h5* which is referenced in the config:

```json
  "inputs": {
    "exc_spikes": {
      "input_type": "spikes",
      "module": "h5",
      "input_file": "inputs/virt_spikes.h5",
      "node_set": "virt"
    }

```

To run:


```bash
$ python run_bionet.py config.simulation_spikes.json
```

Runs a simulation where each biophysical cells recieve spike trains from external cells. It uses a separate network
(network/virt_node*) with synaptic weight/location set in the files network/virt_bio_edges.h5 and 
network/virt_bio_edge_types.csv, with spike trains set in inputs/exc_spike_trains.h5

### Extracellular stimulation
Stimulates the cells by providing extracellular electrode pulses

```json
  "inputs": {
    "Extracellular_Stim": {
      "input_type": "lfp",
      "node_set": "all",
      "module": "xstim",
      "positions_file": "$STIM_DIR/485058595_0000.csv",
      "mesh_files_dir": "$STIM_DIR",
      "waveform": {
        "shape": "sin",
        "del": 1000.0,
        "amp": 0.100,
        "dur": 2000.0,
        "freq": 8.0
      }
    }

```

To run:

```bash
$ python run_bionet.py config.simulation_xstim.json
```

Runs a simualtion where all the cells are stimulated by a extracellular electrode. Extracell stimulation parameters
are set in config_xstim.json


### Spontaneous synaptic stimulation

Directly stimulates the synpase of recurrently connected cells, even if the pre-synaptic cell is inactivated at the 
time

```json
  "inputs": {
    "syn_activity": {
      "input_type": "syn_activity",
      "module": "syn_activity",
      "precell_filter": {
        "model_name": "Scnn1a"
      },
      "timestamps": [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0],
      "node_set": "bio"
    }

```

The timestamps of the spontaneous activatiy are ms from onset of simulation. Will be applied to all synapses in
the **bio** network that are connected to by a model_name="Scnn1a" type cell. To run:

```bash
$ python run_bionet.py config.simulation_spont_activity.json
```
