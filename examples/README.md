# Examples

This is the examples directory for the Brain Modeling Toolkit (bmtk) software package. Here you will find examples of
how to build, simulate, and plot simple brain network models of a variety of different levels-of-resolution using 
bmtk. These examples are toy models for demonstration of how to use bmtk capable of running on a laptop/desktop 
machine (for some examples of scientifically developed models that run on bmtk please see the following link: 
https://alleninstitute.github.io/bmtk/examples.html).


### BioNet (biophysically detailed) models

Each ```bio_*/``` directory uses the BioNet simulator to run morphologically detailed network simulations using the NEURON
simulation tool. 

```bio_components/``` contains external parameter and model files which are shared by most of the BioNet
examples (note: this location can be changed in the each example's ```config.circuit.json``` file). This also includes
a ```bio_components/mechanism/``` directory which require extra compilation for the Allen Institute models. To run 
the BioNet examples one will have to run the following commands to compile the extra neuronal mechanisms:
```bash
$ cd examples/bio_components/mechanisms
$ nrnivmodl modfiles
```

For more information on using BioNet see the following: https://alleninstitute.github.io/bmtk/bionet.html

### PointNet (point-neuron) models

```point_*/``` directories contain examples that use the PointNet simulator to run point-neuron type models, including 
Allen Institute's Generalized Integrate-and-Fire (GLIF) models. Using these examples will require installing the 
NEST simulator. 

```point_components/``` directory contains model files that are shared by many of the different PointNet examples. This
can be changed in each examples' ```config.circuit.json``` files.

For more information on using PointNet see the following: https://alleninstitute.github.io/bmtk/pointnet.html


### PopNet (population firing rates) models

```pop_*/```directories contain examples that use the PopNet simulator to run population level firing-rate model
simulations. This requires installing the DiPDE simulator. 

```pop_components/``` directory contains model files that are shared by many of the different PopNet examples. This
can be changed in each examples' ```config.circuit.json``` files.

For more information on using PopNet see the following: https://alleninstitute.github.io/bmtk/popnet.html


### FilterNet (LNP) models

```filter_*/``` directories contain examples that use the PopNet simulator to run filter models simulations to convert
visual stimuli into spike-trains.

```filter_components/``` directory contains model files that are shared by many of the different FilterNet examples. This
can be changed in each examples' ```config.circuit.json``` files.

For more information on using PopNet see the following: https://alleninstitute.github.io/bmtk/filternet.html


