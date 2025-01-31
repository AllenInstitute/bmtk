Installing the BMTK
===================

**Users are encouraged to register** `here <https://secure2.convio.net/allins/site/SPageServer/?pagename=modeling_tools>`_ 
**to receive updates and other communications, but registration is not required to use the package.** 

The BMTK was developed for use for Python 2.7+ and Python 3.6+. Previous releases can be downloaded from
`here <https://github.com/AllenInstitute/bmtk/releases>`__. The latest code including the newest features and bug fixes
can be found on the `develop branch of our GitHub repo <https://github.com/AllenInstitute/bmtk>`_.

The base installation, which will let you build networks, parse and analyze simulation reports, requires the following
dependencies:

* `numpy <http://www.numpy.org/>`_
* `h5py <http://www.h5py.org/>`_
* `pandas <http://pandas.pydata.org/>`_
* `matplotlib <https://matplotlib.org/>`_
* `jsonschema <https://pypi.python.org/pypi/jsonschema>`_
* `pytest <https://docs.pytest.org/en/latest/>`_ [optional for running unit tests]

All components of the BMTK will work on a single machine, but some parts can take advantage of parallelism against
HPC clusters, and requires the following to be installed:

* `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_


For running network simulations BMTK uses existing software which varies depending on the type of simulation you want
to run. Individual instructions are done for the various requirements of the different simulation engines.


Installing the latest from the source
-------------------------------------
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
The BMTK docker container lets you build and simulate networks without requiring installing all the BMTK prerequisites on
your computer. All you need is the `docker client <https://docs.docker.com/install/>`__ installed on your machine.

::

  $ docker pull alleninstitute/bmtk

There are two main ways of using the bmtk-docker image, either as a command-line application or as a Jupyter Notebook
server

Through the command line
++++++++++++++++++++++++

Go to the directory where your BMTK network-builder and/or network-simulation scripts and supporting files are located
(*local/path*) and run the following in a command-line terminal

::

  $ docker run alleninstitute/bmtk -v local/path:/home/shared/workspace python <my_script>.py <opts>

Due to the way docker works, all files to build/run the network must be within *local/path*, including network files,
simulator components, output directories, etc. If your config.json files references anything outside the working
directory branch things will not work as expected.

**NEURON mechanisms:**
If you are running BioNet and have special mechanisms/mod files that need to be compiled, you can do so by running

::

  $ cd /path/to/mechanims
  $ docker run -v $(pwd):/home/shared/workspace/mechanisms alleninstitute/bmtk nrnivmodl modfiles/


Through Jupyter Notebooks
+++++++++++++++++++++++++
The bmtk-docker image can be run as a jupyter notebook server. Not only will it contain examples and notebook tutorials
for you to run, but you can use it to create new bmtk notebooks. In a command-line run

::

  $ docker run -v $(pwd):/home/shared/workspace -p 8888:8888 alleninstitute/bmtk jupyter


Then open a browser to 127.0.0.1:8888/. Any new files/notebooks should be saved to *workspace* directory, otherwise they
will be lost once the container is closed.


Through Neuroscience Gateway (NSG)
++++++++++++++++++++++++++++++++++
The bmtk can be run through the Neuroscience Gateway for anyone who has an account.

For running multi-core BioNet (NEURON-based) simulations on the NSG please refer to `Readme file <https://github.com/AllenInstitute/bmtk/tree/develop/examples/bio_nsg_template>`_.


Running Simulations
-------------------

Biophysically detailed simulations
++++++++++++++++++++++++++++++++++

Running simulations of biophysically detailed, multi-compartmental neuronal models are done with `BioNet <bionet>`_ which
uses the NEURON simulator, version 7.4 and above. Precompiled version of NEURON can be downloaded and installed from
`here <https://www.neuron.yale.edu/neuron/download/precompiled-installers>`__. Make sure to install the Python bindings.

The precompiled installed version of NEURON may have issues if you are using an Anaconda or Python virtual environment.
Similarly, it will not compile with MPI support so you can't parallelize the simulation. You can find instructions
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
is already built into the BMTK and no extra installation instructions are required.


Population-Level Models
+++++++++++++++++++++++

PopNet will simulate population-level firing rate dynamics using `DiPDE <https://github.com/AllenInstitute/dipde>`_. Instructions
for installing DiPDE can be found `here <http://alleninstitute.github.io/dipde/user.html#quick-start-install-using-pip>`_.
However, we recommend installing DiPDE using anaconda::

  $ conda install -c nicholasc dipde

