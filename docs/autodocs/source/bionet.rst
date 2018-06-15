BioNet
======

BioNet is a high-level interface to `NEURON <http://neuron.yale.edu/neuron/>`_ that facilitates simulations of large-scale neuronal networks on parallel computer architecture. The soure code for BioNet is located in the folder /bmtk/simulator/bionet


Installation
------------
BioNet runs on Python 2.7. If you do not have Python already installed, we recommend installing `Anaconda <https://www.anaconda.com/download/>`_ distribution of Python that comes preloaded with many packages and allows an easy installation of additional packages. 

BioNet requires the following additional python packages:
 * numpy 1.10
 * pandas 0.19.2
 * h5py 2.6
 * jsonschema 2.6.0

To install BioNet either get the latest source-code from the github develop branch:
::
  git clone https://github.com/AllenInstitute/bmtk.git

or `download <https://github.com/AllenInstitute/bmtk/archive/develop.zip>`_ and unzip, and install missing python
dependencies by running from the bmtk/ base directory:
::
  python setup.py install

Installing NEURON
-----------------
In addition, BioNet requires 7.4+ of NEURON simulator. See the `NEURON download documentation <http://www.neuron.yale.edu/neuron/download>`_ for instructions on how to download and install NEURON on your given system.  

If compiling NEURON from the source you will need to provide configuration options (See Appendix in `Hines et al. Frontiers of Neuroinformatics (2009) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2636686/>`_ for details):
 * Make sure to use the '--with-nrnpython' option to allow running NEURON using Python as an interpreter. 
 * If you want to run NEURON in parallel, it requires a version of `MPI <http://www.mpich.org/>`_ to be installed. If you have MPI installed you will configure installation using '--with-paranrn' option. 
 * Furthermore, since BioNet does not require NEURON's GUI functionality, one may install NEURON without GUI support using '--without-x' option. 

Running a test example
----------------------

There are a few examples of how to run BioNet using existing networks located in the directory
docs/examples. There is an example of a small 14 cell network (/14cells) and a slightly larger
network with 450 cells (/450cells). 

The networks utilize models of individual cells, synapse and recording electrodes defined in docs/examples/simulators/biophys_components directory.


Inside each example folder you will find:

`./network` : folder including files describing parameters of the nodes (cells) and edges (connections) making up the network

`config.json` : configuration file incuding paths to files describing simulated network, output files as well as run parameters

`run_bionet.py` : main python script which calls BioNet to run simulation

`set_cell_params.py, set_syn_params.py, set_weights.py` : modules which describe how models of cells, synapses and connection weights are created.


These examples use biophysically-detailed models of individual cells, and thus require additional NEURON channel
mechanisms describing dynamics of ionic channels. To compile these NEURON mechanims go to the subdirectory docs/examples/simulators/biophys_components/mechanisms and run the NEURON command
::
   nrnivmodl modfiles/

To run the examples in their respective subdirectories, you can run a full BioNet simulation on a single core by running
the command:
::
  python run_bionet.py config.json

When simulation is completed you will see a message “Simulation completed” and you should have an output directory ./output

Tutorials
---------

Please see the `BioNet examples tutorial <./bionet_tutorial.html>`_ for a more detailed explanation of how to simulate example networks.






