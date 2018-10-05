PointNet
========

Pointnet is a simulation engine that utilizes `NEST <http://www.nest-simulator.org/>`_ to run large-scale point
neuron network models.

Features
--------
* Run the same simulation on a single core or in parallel on an HPC cluster with no extra programming required.
* Supports any spiking neuron models (rates models in development) available in NEST or with user contributed modules.
* Records neuron spiking, multi-meter recorded variables into the optimized SONATA data format.


Installation
------------
PointNet supports both Python 2.7 or Python 3.6+, and also requires NEST 2.11+ to be installed. See our
`Installation instructions <installation>`_ for help on installing NEST and the BMTK.



Documentation and Tutorials
---------------------------
Our `github page <https://github.com/AllenInstitute/bmtk/tree/develop/docs/tutorial>`__ contains a number of jupyter-notebook
tutorials for using the BMTK in general and PointNet specific examples for:
* `Building and simulating a multi-population, heterogeneous point networks <https://github.com/AllenInstitute/bmtk/blob/develop/docs/tutorial/05_pointnet_modeling.ipynb>`_.



Previous Materials
++++++++++++++++++
The following are from previous tutorials, workshops, and presentations; and may not work with the latest version of the BMTK.
* CNS 2018 Workshop: `notebooks <https://github.com/AllenInstitute/CNS_2018_Tutorial/tree/master/bmtk>`__
* Summer Workshop on the Dynamic Brain 2018: `notebooks <https://github.com/AllenInstitute/SWDB_2018/tree/master/DynamicBrain/Modeling>`__.


Examples
--------
The AllenInstitute/bmtk repo contains a number of PointNet examples, many with pre-built networks and can be immediately ran. These
tutorials will have the folder prefix *point_* and to run them in the command-line simply call::

  $ python run_pointnet.py config.json

or to run them on multiple-cores::

  $ mpirun -n $NCORES python run_pointnet.py config.json

Current examples
++++++++++++++++
* `point_120cells <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/point_120cells>`_ - A small network of 120 recurrently connected point neurons receiving synaptic input from an external network of "virtual" cells (i.e. spike-generators).
* `point_450cells <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/point_450cells>`_ - A modest heterogeneous network of 450 cells (sampled from a much large network of a mouse cortex L4).

