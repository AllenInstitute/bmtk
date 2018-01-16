The Builder
===========

The Builder allows for the building and saving of brain-network models. It allows for the building of various types
of networks, from biophysically detailed networks of neurons and synapses to networks of interconnected neuronal
populations. Networks are saved into a standarized format that can be used by a variety of different simulators and
visualization tools.


Installation:
-------------

The Builder currently runs onf Python 2.7. To install the requirements using using python setup tools run from a
command terminal
::

    python setup.py install


Examples
--------

Examples of creating different types of networks can be found in the documentation under examples/builder/. To run one
of the examples go into the subdirectory and run
::
  python build_network.py

The network(s) will be saved to the output/ folder


Tutorial
--------

Please see our tutorials for how to build and run networks using the bmtk:
https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/00_introduction.ipynb