# Spontaneous Synaptic Activity

This example is for a special functionality to bmtk that allows for certain synapses to activate even when the cells
themselves are not spiking. To implement the *config.json* file contains the following section

```json
  "inputs": {
    "syn_activity": {
      "input_type": "syn_activity",
      "module": "syn_activity",
      "precell_filter": {
        "pop_name": "e4Scnn1a"
      },
      "timestamps": [200.0],
      "node_set": "v1"
    }
  },
```

The important parameters are **precell_filter**, which used to distinguish what synapses should file depending on their
presynaptic cell type, and **timestamps** is the time(s) in milliseconds which the synapses are activated. In the above
example connections with presynaptic cell population (see v1_node_types.csv) "e4Scnn1a" should all activate at 200 ms
into the simulation.

All synapses that don't fire spontaneously will have their synapses zeroed out (syn_weight == 0).


#### Implementation
There are two new classes for implementing this behavior; *bmtk.simulation.bionet.biocell.BioCellSpontSyn* and
*bmtk.simulator.bionet.pointprocesscell.PointProcessCellSpontSyns*

