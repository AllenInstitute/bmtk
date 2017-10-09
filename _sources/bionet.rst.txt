BioNet
======

BioNet is a simulation engine that utilizes `NEURON <http://neuron.yale.edu/neuron/>`_ to run large-scale biophysically
detailed network simulations.

Installation
------------
BioNet runs on Python 2.7 and will require addtional python packages numpy, h5py, pandas and jsonschema. To install
either get the latest source-code from the github develop branch
::
  git clone https://github.com/AllenInstitute/bmtk.git

or `download <https://github.com/AllenInstitute/bmtk/archive/develop.zip>`_ and unzip, and install missing python
dependencies by running from the bmtk/ base directory
::
  python setup.py install

Installing NEURON
-----------------
In addition, BioNet requires 7.4+ of NEURON simulator. See the `NERUON download documentation<http://www.neuron.yale.edu/neuron/download>`_
for instructions how on how to download and install NEURON on your given system.

BioNet requires the NEURON python libraries be included and installed on the local python 2.7 intrepreter. If
compiling NEURON from the source make sure to use the --with-nrnpython option. MPI (--with-paranrn option) is not
required although it is highly recommended for running very large networks.

Examples
--------
There are multiple examples of how to run BioNet using existing networks located in the source-code directory
docs/examples/simulators/bionet. There is an example of a small 14 cell network (14cells/) and a slightly larger
network with 450 cells (450cells).

Both these examples uses models taken from the Allen Cell-Types Database, and thus require additional NEURON channel
mechanics. To build and install these ion channels go to the subdirectory docs/examples/simulators/bionet/components/mechanisms and
run the NEURON command
::
   nrnivmodl modfiles/

To run the examples in their respective subdirectories, you can run a full BioNet simulation on a single core by running
the command
::
  python run_bionet.py config.json

Or to run on parralized cluster with MPI setup
::
  mpirun -np $NCORES nrniv -mpi -python run_bionet config.json

**Setup files**

 * config.json - configuration file that contains everything needed to setup and run a simulation.
 * network/ - directory of network files (node and edge files) to describe the various models being simualted. Please see the Builder documentation for how these files were built and modified.
 * components/ - includes morphology files, cell and synaptic models, and additional mechanisms required to run a full simulation.


**output**

After the simulation has finished, the results are placed into the output/ folder. The output includes
 * spikes.txt - contains spikes of the (non-virtual) network cells, ordered by gid then spike-time.
 * cellvars/*N*.h5 - HDF5 file containing information about the membrane potential and calcium traces for cell # *N*



