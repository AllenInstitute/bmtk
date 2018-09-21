Installing the BMTK
===================

The bmtk was developed for use for Python 2.7+ and Python 3.6+. Previous releases can be downloaded from
`here <https://github.com/AllenInstitute/bmtk/releases>`__. The latest code including newest features and bug fixes
can be found on the `develop branch of our github repo <https://github.com/AllenInstitute/bmtk>`_.

The base installation, which will let you build networks, parse and analyze simulation reports, require the following
dependencies:

* `numpy <http://www.numpy.org/>`_
* `h5py <http://www.h5py.org/>`_
* `pandas <http://pandas.pydata.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `jsonschema <https://pypi.python.org/pypi/jsonschema>`_
* `pytest <https://docs.pytest.org/en/latest/>`_ [optional for running unit tests]

All components of the bmtk will work on a single machine, but some parts can take advantage of parallelism againsts
HPC clusters, and requires the following to be installed:

* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_


For running network simulations bmtk uses existing software which varies depending on the type of simulation you want
to run. Individual instructions are done for the various requirements of the different simulation engines.


Installing latest from source
-----------------------------
::

  $ git clone https://github.com/AllenInstitute/bmtk.git
  $ cd bmtk
  $ python setup.py install


Installing with pip
-------------------
::

  $ pip install bmtk


Installing with conda
---------------------
::

  $ conda install -c kaeldai bmtk


Using the Docker Image
----------------------

More Info to come!


Running Simulations
-------------------

Biophysically detailed simulations
++++++++++++++++++++++++++++++++++

Running simulations of biophysically detailed, multi-compartmental neuronal models is done with `BioNet <bionet>`_ which
uses the NEURON simulator, version 7.4 and above. Precompiled version of NEURON can be downloaded and installed from
`here <https://www.neuron.yale.edu/neuron/download/precompiled-installers>`__. Make sure to install the Python bindings.

The precompiled installed version of NEURON may have issues if you are using a Anaconda or Python virtual environment.
Similarly it will not compile with MPI support so you can't parallelize the simulation. You can find instructions
for compiling NEURON for `Linux <https://www.neuron.yale.edu/neuron/download/compile_linux>`_,
`Mac <https://www.neuron.yale.edu/neuron/download/compilestd_osx>`_, and
`Windows <https://www.neuron.yale.edu/neuron/download/compile_mswin>`_.

* **NOTE** - BioNet does not use the NEURON GUI (iv package) and we recommend you compile with the --without-iv option


Point-Neuron simulations
++++++++++++++++++++++++

The PointNet simulation engine is responsible for running point-neuron networks using the `NEST simulator <http://www.nest-simulator.org/>`_,
version 2.11 and above. Instructions for installing NEST can be found `here <http://www.nest-simulator.org/installation/>`__.


Filter Models
+++++++++++++

FilterNet is the simulation engine responsible for simulating firing rate responses to stimuli onto the visual fields. It
uses a piece of simulation software called LGNModel developed at the Allen Institute for Brain Science. Luckily, LGNModel
is already built into the bmtk and no extra installation instructions are required.


Population Level Models
+++++++++++++++++++++++

PopNet will simulate population level firing rate dynamics using `DiPDE <https://github.com/AllenInstitute/dipde>`_. Instructions
for installing DiPDE can be found `here <http://alleninstitute.github.io/dipde/user.html#quick-start-install-using-pip>`_.
However we recommend installing DiPDE using anaconda::

  $ conda install -c nicholasc dipde

