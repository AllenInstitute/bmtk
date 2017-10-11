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

Examples
--------
There are a few examples of how to run BioNet using existing networks located in the directory
docs/examples/simulators/bionet. There is an example of a small 14 cell network (/14cells) and a slightly larger
network with 450 cells (/450cells). 

The network utilize models of individual cells, synapse and recording electrodes defined in docs/examples/simulators/bionet/components/mechanisms directory.

Inside each example folder you will find:

**Model description**

`config.json` : configuration file incuding paths to files describing simulated network, output files as well as run parameters

`./network` : folder including files describing parameters of the nodes (cells) and edges (connections) making up the network

**Python scripts**

`run_bionet.py` : main python script which calls BioNet to run simulation

`set_cell_params.py, set_syn_params.py, set_weights.py` : modules which describe how models of cells, synapses and connection weights are created.


These examples use biophysically-detailed models of individual cells, and thus require additional NEURON channel
mechanisms describing dynamics of ionic channels. To compile these NEURON mechanims go to the subdirectory docs/examples/simulators/bionet/components/mechanisms and run the NEURON command
::
   nrnivmodl modfiles/

To run the examples in their respective subdirectories, you can run a full BioNet simulation on a single core by running
the command:
::
  python run_bionet.py config.json

Or to run in parallel with MPI setup on $NCORES:
::
  mpirun -np $NCORES nrniv -mpi -python run_bionet config.json



**Simulation output**

BioNet allows saving simulation output in blocks while simulation is still running, giving users an ability to check and analyze intermediate output. During the run you will see some output reporting on the progress of a simulation. When simulation completed you will see a message "Simulation completed".

The output directory includes:
 * spikes.h5 : HDF5 file contains spikes of the simulated cells.
 * cellvars/N.h5 : HDF5 file containing time series recordings of somatic variables  (e.g., somatic voltage, [Ca++]) for cell with node_id=N. 
 * config.json : a copy of configuration for the record keeping
 * log.txt : run log file including time-stemped information about the progress of a simulation. 



Simulating your network models
------------------------------

To run simulations of your network with BioNet, you will first need to provide a pre-built network in the format understood by BioNet. We recommend using BMTK's builder tools, but you may also use your own scripts or a third party tool to build a network. As a start we suggest to modify the existing network examples as a quick way of customizing network models and then build your own model following `builders examples tutorial <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/builder/bionet_14cells>`_.

When you have your custom model built, you will need to specify in the your configuration file the paths to the network, components as well as simulation run parameters.

Just as in the above examples, your run folder should include Python modules: set_cell_params.py, set_syn_params.py, set_weights.py specifying how models of cells, synapses and connection weights are created as well as a main python script. 

When running different simulations you will rarely need to modify the main Python script running BioNet. Instead, you will commonly need to modify paths to network files or run parameters in the configuration file  to instruct BioNet which model to run and how to run it. Please refer to the `configuration file tutorial <./bionet_config.html>`_ for details.



