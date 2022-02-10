# Burning Ring of Integrate-and-Fire

Network contains a single morphologically detailed neuron surrounded by a ring of (default 6) point_processes
intfire neurons. Each point_neuron independently synapses onto the biophysical neuron, providing feed-forward
stimulation of the biophysical neuron as the point_neurons spontaneously fire. 

The main network is called "v1" with files **v1_nodes** and **v1_v1_edges**. There is also a separate network of
**virtual** cells that provided the spike-trains to the point_process neurons. 

