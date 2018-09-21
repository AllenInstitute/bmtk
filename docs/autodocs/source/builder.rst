The Network Builder
===================

To run a simulation or visualizing a neural network the Brain Modeling Toolkit (bmtk) requires network files. This
workflow of using intermediate files, rather than building and simulating an entire network in memory, is done to
improve performance and memory, helps with iterative modeling, and allows for sharing and reproducibility. Before
running a simulation you will first need to obtain existing network model files, or alternativly use the bmtk
**Builder** submodule to create them from scratch.

The Builder uses the same python inteface for creating networks of various varieties including multi-compartment neurons,
point neuron, and population levels-of-resolution. It can support any model structure that is capable of being described
by a directed graph of nodes (neurons, populations, regions) and edges (synapses, junctions, fibre tracts). It is also
simulator agnostic, modelers can choose whatever attributes and properties they require to represent their model
(*However to simulate a given network certain attributes will be required, depending on the underlying simulator*).

The bmtk builder will build and save network files using `SONATA <https://github.com/AllenInstitute/sonata>`_; a highly
optimized data format for representing large-scale neural networks.



Installing the Builder
-----------------------

The Builder sub-module is automatically included in the bmtk package. For instructions please refer to the `installation page <installation>`__.



Tutorials and Guides
--------------------

For a general overview of how to use the Builder please see the following tutorial:
   https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/NetworkBuilder_Intro.ipynb


Similarly there are also a number of tutorials showing how to build (and simulate) networks of different levels:
   https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/00_introduction.ipynb



Examples
--------

A good way of getting started with building networks is to start with an existing example. The AllenInstitute/bmtk github
repo `includes a number of examples <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples>`__. Inside most
examples you should find a *build_network.py* which shows how to build the network for that given simulation (and the
output of the Builder will be saved in the *network/* directory).


Many of the `examples in the SONATA repo <https://github.com/AllenInstitute/sonata/tree/master/examples>`__ were also built using bmtk and includes *build_network.py* scripts
