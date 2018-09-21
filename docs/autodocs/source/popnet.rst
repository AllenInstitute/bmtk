PopNet
======

The PopNet simulation engine uses the `DiPDE simulator <http://alleninstitute.github.io/dipde/>`_ to simulate firing
rate dynamics of connected population of cells.

Installation
------------
PopNet runs on Python 2.7 and will require addtional python packages numpy, h5py, pandas and jsonschema. To install
either get the latest source-code from the github develop branch
::

  git clone https://github.com/AllenInstitute/bmtk.git

or `download <https://github.com/AllenInstitute/bmtk/archive/develop.zip>`_ and unzip, and install missing python
dependencies by running from the bmtk/ base directory


Installing DiPDE
----------------
PopNet will require DiPDE to run a simulation. To install DiPDE you can use pip
::

   pip install git+https://github.com/AllenInstitute/dipde.git --user

Alteratively using Anaconda
::

   conda install -c nicholasc dipde

Examples
--------
Examples for running PopNet are located in the source directory docs/examples/simulator/popnet/. Each subdirectory contains
a different example with the following structure:

* network/ - files used to describe the networks being simulated
* models/pop_models/ - configuration files that setup different populations in the network.
* models/synaptic_models/ - configuration files that specify the connection between different populations.

To run a simulation
::

   python run_popnet.py config.json

**Output**

After the simulation is finished, firing rates are saved into output/spike_rates.txt. The file follows the structure of
space seperated:

   population-id  time  firing-rate