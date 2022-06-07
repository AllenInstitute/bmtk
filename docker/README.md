# docker-bmtk

With Docker, you can test and run the bmtk without having to go through the hassle of installing all the prerequisites (incl.
NEURON, NEST, DiPDE, etc). All you need is the Docker client installed on your computer. You can use the bmtk Docker
container through the command line to build models and run simulations. Or you can use it as a Jupyter Notebook container
to test out existing tutorials/examples, or create new Notebooks yourself

_Note_: You will not be able to utilize parallelization support (MPI) if running bmtk through Docker. Similarly, you can
expect memory issues and slowness for larger networks. For building and simulating large networks we recommend installing
bmtk and the required tools natively on your machine.

## Getting the Image

You can pull the bmtk container from DockerHub

```bash
  $ docker pull alleninstitute/bmtk
```

Or to build the image from the bmtk/docker directory

```bash
  $ docker build -t alleninstitute/bmtk .
```

## Running the BMTK

### Through the command-line

To run a network-build or simulation-run bmtk script using the docker container, go to the directory containing your
python script and any necessary supporting files:

```bash
  $ docker run alleninstitute/bmtk -v $(pwd):/home/shared/workspace python <my_script>.py <opts>
```

**NOTE**: All files must be under the directory you are running the command; including network, components, and output
directories. If your config.json files references anything outside the working directory branch things will not work
as expected.

#### NEURON Mechanisms

If you are running BioNet and have special mechanisms/mod files that need to be compiled, you can do so by running:

```bash
  $ cd path/to/mechanims
  $ docker run -v $(pwd):/home/shared/workspace/mechanisms alleninstitute/bmtk nrnivmodl modfiles/
```

### Through Jupyter Notebooks

To run a Jupyter Notebook server:

```bash
  $ docker run -v $(pwd):/home/shared/workspace -p 8888:8888 alleninstitute/bmtk jupyter
```

Then open a browser to 127.0.0.1:8888/. Any new files and/or notebooks that you want to save permanently should
be created in the workspace folder, otherwise, the work will be lost when the server is stopped.
