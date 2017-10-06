
Below instructions apply for installing BioNet and running simulationson Linux platform, although it was also tested on MacOS platfrom:

BioNet package is a component of the larger Brain Modeling ToolKit package https://github.com/AllenInstitute/bmtk
To install BioNet you will install the Brain Modeling Toolkit by clicking on "Clone or download" button and download the zip file to the desired folder. 
 
cd to that folder and unzip the file will create bmtk-develop folder:

To compile
```bash
$ unzip bmtk-develop
```

You will also need to add a path to bmtk-develop folder to you PYTHOPATH variable in your bashrc file.

BioNet package requires Python 2.7 (www.python.org). If you do not have Python installed we recommend installing Anaconda distribution of Python (https://www.anaconda.com/download/) that comes preloaded with many packages and easy installation of additional packages.

BioNet has the following Python package dependancies which may need to be installed prior to running the code

numpy 1.10 
pandas 0.19.2
h5py 2.6 
jsonschema

***Kael, what should I write about setup.py?

If you just installed Python then you will also need to include a path to Python's to your PATH variable and site packages to PYTHONPATH. For instance, for anaconda you will need to add patht to anaconda2/bin to your PATH and anaconda2/lib/python2.7/site-packages to your PYTHOPATH.

At this stage try running python and importing Brain Modeling Toolkit as follows:

```bash
$ python
>>> import bmtk
>>> exit()
```

Additionally, BioNet requires NEURON 7.4 simulator (http://www.neuron.yale.edu/neuron/download/). Detailed installation instructions may be found in the Appendix in Hines et al. (2007). Since BioNet does not require NEURON's GUI functionality, one may install NEURON without GUI support provided by InterViews. For convenience we provide installation instructions of NEURON on Linux platform without InterViews below:

create an neuron directory:

```bash
$ mkdir neuron
$ cd neuron
$ download and unzip a nrn-7.4.rel-1370.tar.gz version of neuron:
$ wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.4/v7.4.rel-1370/nrn-7.4.rel-1370.tar.gz
$ tar xf nrn*gz
$ cd nrn-7.4
```

Other versions of NEURON may be found here http://www.neuron.yale.edu/ftp/neuron/versions/ and similarly installed.

If you want to run parallel NEURON in parallel, it requires a version a version of MPI to be installed, then you will configure installation as follows: 

change to the newly created nrn-7.4 directory and configure installation:

```bash
$ ./configure --prefix=`pwd` --with-paranrn --with-nrnpython --without-x have_cython=no BUILD_RX3D=0
```

otherwise remove --with-paranrn from configuration options

```bash
finally build neuron binaries:
make -j 6 install
```


you will need to add path to neuron/nrn-7.4/x86_64/bin to your PATH variable and also
add path to /home/sergeyg/neuron/nrn-7.5/lib/python to your PYTHONPATH variables

If installed correctly you should be able to do the following:

```bash
$ nrniv -python
>>> import bmtk
>>> import neuron
>>> exit()
```

Now you are ready to run examples:

To run examples go to the folder with installed bmkt-develop package and cd to the examples folder for BioNet 
cd docs/examples/simulator/bionet/


There you will find a 4 folders with examples of pre-built and ready to run networks
Prior to running you will need to compile NEURON mechanisms in the components folder:

```bash
$ cd  ./components/mechanisms/
$ nrnivmodl modfiles
```

which will create /x86 folder

now cd to the specific example such as 14cells

inside each example folder you will find

run_bionet.py : main python script which calls BioNet to run simulation
config.json : configuration file incuding paths to files describing simulated network as well as run parameters
./network : folder including files describing parameters of the nodes and edges making up the network

set_cell_params.py set_syn_params.py set_weights.py : modules which describe how models of cells, synapses and connection weights are created.


To run simulation on a single processor:
nrniv -python run_bionet.py config.json
If you have MPI installed you may run simulation in parallel (e.g. on 4 processor) as follows:

```bash
$ mpirun -np 4 nrniv -mpi -python run_bionet.py config.json
```


The code will first check whether the folders listed in the configuration file are are present and will provide an informative error message otherwise. If you receive an error message, check the paths in the configuration file.
Upon succesful validation of configuration BioNet will proceed to load the network, simulate the activity and save the results to files in the output directory.

In the output directory you will find:
spikes.h5: hdf5 file containing spike information.
spikes.txt: same as above, but using ascci file for users not familiar with hdf5.
./cellvars: an folder including hdf5 files with simulation output for individual select cells. The desired varibles and ids of cells (node_ids) to output are specified in the configuration file:

	"save_cell_vars": ["v", "cai"] and ).

	  "groups": {
		"save_vars": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	  },

config.json: a copy of configuration file for the record keeping
log.txt: run log file including time-stemped information about simulation run progress. 

