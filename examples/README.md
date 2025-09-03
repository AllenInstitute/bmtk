# Examples

The following directory contains examples of running various parts of the Brain Modeling Toolkit (BMTK), including all the necessacary files to build a network, simulate it with one or more types of stimulation, and record and plot results. Each directory highlights a different type of model or a different features available in BMTK. 

The majority of the examples are small toy examples that can be readily ran on most laptops or desktops. The minimum they requires BMTK, but depending on the module may have additional requirements.


## Directory Structure

Most of the examples have a prefix based on which simulation engine is being used to run the model.

- **bio_\*/** - Examples of biophysically detailed network models that run using [BioNet](https://alleninstitute.github.io/bmtk/bionet.html). 
- **point_\*/** - Examples of point-neuron network models that run using [PointNet](https://alleninstitute.github.io/bmtk/pointnet.html)
- **pop_\*/** - Examples of populations based rates models that run using [PopNet](https://alleninstitute.github.io/bmtk/popnet.html)
- **pop_\*/** - Examples that uses [FilterNet](https://alleninstitute.github.io/bmtk/filternet.html) to convert visual or auditory stimuli into spike-models based on spatio-temporal statistics.

Other important directories

- **bio_components/** - External files shared by BioNet models and simulations (Morphologies, parameters, etc.).
- **point_components/** - External files shared by PointNet models and simulations
- **pop_components/** - External files shared by PopNet models and simulations

## Running Simulations

### BioNet

**(Prerequisite) Compiling NEURON mechanisms**

The components for the BioNet examples are located in ../examples/bio_components. If the NEURON mechanisms have not already been compiled, the following should compile the NEURON mechanisms and place them in another folder in /mechanisms.

```bash
$ cd ../bio_components/mechanisms
$ nrnivmodl modfiles 
$ cd -
```
Failure to compile the mechanisms results in an error such as:
```
**Warning**:  NEURON mechanisms not found in ./../bio_components/mechanisms.
              [...]
              ValueError: argument not a density mechanism name
```
---


To run a full simulation of the network on a single using the default simulation configuration, run the following on a command line:

```bash
$ python run_bionet.py
```

or to use a different SONATA configuration file (eg `<SONATA_CONFIG>.json`)

```bash
$ python run_bionet.py <SONATA_CONFIG>.json
```

If using a machine or cluster with mutiple cores you may use `mpirun` (if available on machine) to run the simulation in `N` cores with the following command:

```bash
$ mpirun -np <N> nrniv -mpi -python run_bionet.py <SONATA_CONFIG>.json
```

When simulation has completed it will create an *output* folder with logs, simulated spike trains, and any other recorded variables as set in the `<SONATA_CONFIG>`


### PointNet

Executing a simulation inside the model directory can be done in the command line using either

```bash
$ python run_pointnet.py 
```

Which will run the default simulation configurations. Or to run a different simulation setup you can specify a specific `<SONATA_CONFIG>.json` file path:

```bash
$ python run_pointnet.py <SONATA_CONFIG>.json
```

If using a machine or cluster with mutiple cores you may use `mpirun` (if available on machine) to run the simulation in `N` cores with the following command:

```bash
$ mpirun -np <N> python run_pointnet.py <SONATA_CONFIG>.json
```

When simulation has completed it will create an *output* folder with logs, simulated spike trains, and any other recorded variables as set in the `<SONATA_CONFIG>`

### PopNet

Executing a simulation inside the model directory can be done in the command line using either

```bash
$ python run_popnet.py 
```

Which will run the default simulation configurations. Or to run a different simulation setup you can specify a specific `<SONATA_CONFIG>.json` file path:

```bash
$ python run_popnet.py <SONATA_CONFIG>.json
```

If using a machine or cluster with mutiple cores you may use `mpirun` (if available on machine) to run the simulation in `N` cores with the following command:

```bash
$ mpirun -np <N> python run_popnet.py <SONATA_CONFIG>.json
```

When simulation has completed it will create an *output* folder with logs, simulated spike trains, and any other recorded variables as set in the `<SONATA_CONFIG>`


## Updating Simulation Parameters

BMTK uses the *config.simulation_\*.json* files to determine simulation parameters like run-time, time delta, stimulus, recorded varaibles, and a number of other important factors. These are simple json files that can be edited with most text editors. 

Please see our [BMTK User Guide](https://alleninstitute.github.io/bmtk/user_guide.html) or the [SONATA Developers Guide](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md) for a description of the configuration format and a list of features and attributes available in BMTK.