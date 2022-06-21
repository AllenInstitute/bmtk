# The Brain Modeling Toolkit

A software development package for building, simulating, and analyzing large-scale networks of different levels of resolution.

Registering allows us to communicate with BMTK users and is encouraged, but not required: [registration link](https://secure2.convio.net/allins/site/SPageServer/?pagename=modeling_tools).

See the paper about BMTK: [link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008386).

Please cite BMTK as follows:

Dai et al. Brain Modeling Toolkit: An open-source software suite for multiscale modeling of brain circuits. PLoS Comput Biol 16(11): e1008386. https://doi.org/10.1371/journal.pcbi.1008386

## Level of Support

We are releasing this code to the public as a tool we expect others to use. Questions concerning bugs and related issues are welcomed. We expect to address them promptly and pull requests will be vetted by our staff before inclusion.

## Quickstart

bmtk requires Python 2.7 or 3.5+, plus [additional python dependicies](https://alleninstitute.github.io/bmtk/index.html#base-installation). To install with
base requirements from a command-line:

```bash
 $ git clone https://github.com/AllenInstitute/bmtk.git
 $ cd bmtk
 $ python setup.py install
```

There are examples of building models and running simulations located in docs/examples/. Some of the simulation engines may require additional requirements to run.

##### Tests

There are a collection of unit tests in `bmtk.tests` which can be run using pytest

```bash
  $ cd bmtk
  $ py.test
```

## Documentation

[User Guide](https://alleninstitute.github.io/bmtk/)

- [Building network models](https://alleninstitute.github.io/bmtk/builder.html)
- [Running biophysical simulations](https://alleninstitute.github.io/bmtk/bionet.html)
- [Running point-neuron simulations](https://alleninstitute.github.io/bmtk/pointnet.html)
- [Running population-level simulations](https://alleninstitute.github.io/bmtk/popnet.html)

Copyright 2017 Allen Institute
