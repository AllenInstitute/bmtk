Running BioNet on the Neuroscience Gateway (NSG) Portal
=======================================================

The following directory is a template for running BioNet (NEURON based) simulations on NSG. Before running a NSG
simulation please follow the below steps to create a working input file.

### 1. Add network files.

Place all the SONATA network files in the **network** directory. Open up **config.json** with a text editor and update the __networks__ section to point to the added files.


### 2. Inputs

SONATA spike train files, used to drive the network, should be placed in the **spikes_inputs** directory, and the __inputs__ section of **config.json** should be updated to reflect the files.


### 3. Model files

Hoc, NML, swc, and json files used to model individual cells and connections should be add to their respective subdirectories under **biophys_components**.


### 4. Creating the zip file.

Once the directory has been updated, zip up this directory.


### 5. Running on NSG

Upload the zipped directory to your NSG Data folder. To run bmtk simulation create a new task.
 * Under __Select Tools__ choose either "NEST7.4 using Python on Comet" or "Python on Comet".
 * Under __Set Parameters__ make sure the "Main Input Filename" is set to **run_bionet.py**
