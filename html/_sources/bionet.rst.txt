BioNet
======

BioNet is a high-level interface to `NEURON <http://neuron.yale.edu/neuron/>`_ that facilitates simulations of large-scale
networks of multicompartmental neurons. It is built on top of Python, but allows users to run simulations quickly
often with little-to-no programming.



Features
--------
* Automatically integrates MPI for parallel simulations without the need of extra coding.
* Supports models and morphologies from the Allen `Cell-Types Database <http://celltypes.brain-map.org/data>`_, as well
as custom hoc and NeuroML2 cell and synapse models.
* Use spike-trains, synaptic connections, current clamps or even extracellular stimulation to drive network firing.
* Can simulate extracelluarl field recordings.



Installation
------------
BioNet supports both Python 2.7 or Python 3.6+, and also requires NEURON 7.4+ to be installed. See our
`Installation instructions <installation>`_ for help on installing NEURON and the BMTK.



Documentation and Tutorials
---------------------------
Our `github page <https://github.com/AllenInstitute/bmtk/tree/develop/docs/tutorial>`__ contains a number of jupyter-notebook
tutorials for both building multi-compartmental neural networks and simulating them with BioNet, including ones specially
how to:
1. Simulate cell(s) with a `current clamp <https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/01_single_cell_clamped.ipynb>`_.
2. Use synaptic inputs to stimulate `single <https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/02_single_cell_syn.ipynb>`_ and
`multi-cell <https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/03_single_pop.ipynb>`_ networks.
3. Build and simulate `multi-population heterogeneous networks <https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/04_multi_pop.ipynb>`_.


Previous Materials
++++++++++++++++++
The following are from previous tutorials, workshops, and presentations; and may not work with the latest version of the bmtk.
* CNS 2018 Workshop: `notebooks <https://github.com/AllenInstitute/CNS_2018_Tutorial/tree/master/bmtk>`__
* Summer Workshop on the Dynamic Brain 2018: `notebooks <https://github.com/AllenInstitute/SWDB_2018/tree/master/DynamicBrain/Modeling>`__.


Examples
--------
The AllenInstitute/bmtk repo contains a number of BioNet examples, many with pre-built networks and can be immediately ran. These
tutorials will have the folder prefix *bio_* and to run them in the command-line simply call::

  $ python run_bionet.py config.json

or to run them on multiple-cores::

  $ mpirun -n $NCORES nrniv -mpi -python run_bionet.py config.json

Current examples
++++++++++++++++
* `bio_14cells <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/bio_14cells>`_ - A small network of 14 recurrently connected multi-compartment and point-process cells receiving synaptic input from an external network of "virtual" cells (i.e. spike-trains).
* `bio_450cells <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/bio_450cells>`_ - A modest heterogeneous network of 450 cells (sampled from a much large network of a mouse cortex L4).
* `bio_450cells_exact <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/bio_450cells_exact>`_ - Similar to bio_450cells but with synaptic locations precisely stated (In the other examples the simulator will randomly assign synaptic locations at run-time given a predefined range).
* `bio_stp_models <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/bio_stp_models>`_ - An example using Allen Institute custom STP-based synapse models.
* `bio_basic_features <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/bio_basic_features>`_ - Examples how how to stimulate networks using a variety of methods.

