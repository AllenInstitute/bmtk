# Exc-to-Inh simple network with synapses

The following contains a simple 2-neuron network, sampled from the Mouse V1 model, to help with visualization of the
synaptic locations.



## Files

The nodes and edges are saved in the network/ directory using the SONATA format (similar to the V1 network that
has already been visualized). There is also a network/csv/ directory with the nodes and edges in a flat csv table, these
are not part of the SONATA format - but may come in handy for debugging.

The network_config.json file is a SONATA configuration file which contains the location of not only the SONATA edges
and nodes files, but the swc files which are referenced in the nodes' "morphology" column.



## Synapse locations

In the "v1_v1_edges.h5" there are two columns "afferent_swc_id" and "afferent_swc_pos" which can be used to determine
the post-synaptic location on the target neuron. afferent_swc_id will contain an integer with is a direct reference
to the "id" (first) column in the target neuron's *.swc file

The afferent_swc_pos column will contain a floating-point value 0.0 <= x < 1.0, that can be used to determine how
close the actual synapse is to the swc_id row. If the swc_pos is 0.0 then the synapse is located at the exact same
coordinates specified. A value closer to 1.0 means the synaptic location is closer to that row's parent/pid.

(NOTE: For segmented compartmental models the swc_pos is not actually used)
