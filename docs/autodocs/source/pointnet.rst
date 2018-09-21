PointNet
========

Pointnet is a simulation engine that utilizes `NEST <http://www.nest-simulator.org/>`_ to run large-scale point
neuron network models.

Installation
------------
PointNet runs on Python 2.7 and will require addtional python packages numpy, h5py, pandas and jsonschema. To install
either get the latest source-code from the github develop branch
::

  git clone https://github.com/AllenInstitute/bmtk.git

or `download <https://github.com/AllenInstitute/bmtk/archive/develop.zip>`_ and unzip, and install missing python
dependencies by running from the bmtk/ base directory
::

  python setup.py install

Installing NEST
---------------
In addition PointNet requires NEST 2.12.0 python API to run a network simulation. Please see the `NEST installation documentation <http://www.nest-simulator.org/installation/>`_
for further instructions. PointNet does not require MPI to run a simulation, however if it is enabled in NEST then
PointNet can utilized MPI to run simulation on a cluster.

Examples
--------
There is an example simulation of a small 120 cell network located in the source directory docs/examples/pointnet/120cells/.

The network/ directory contains a small example network with nodes and edges files, and was built using the network
builder. The config.json file contains run-time and network setup parameters, including location of cell and synaptic
model files.

To run a simulation in a single processor, open a command prompt and run
::

   python run_pointnet.py config.json

Or to run a parallized simulation with MPI setup
::

   mpirun -np $NCORES python run_pointnet.py config.json

**Output**

Spike-trains of the simulation are saved in the output/spikes.txt file, with two columns time, cell-gid.

