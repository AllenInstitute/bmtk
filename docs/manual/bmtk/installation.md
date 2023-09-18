[Back to ToC](/docs/manual/README.md)
# [Installation guide](https://alleninstitute.github.io/bmtk/installation.html)

## BMTK

BMTK requires Python 2.7 or 3.5+, plus [additional python dependencies](#dependencies) (i.e. python packages that are required for BMTK to function properly). There are three ways to install BMTK with base requirements from a command-line:

- Using your favourite python package manager (pip or conda)
    ```bash
    $ pip3 install bmtk
    ```
     OR
    ```bash
    $ conda install -c kaeldai bmtk
    ```
  - Both pip and conda should automatcally download the necessary python [dependencies](#dependencies).
  - However, you will have to download any tutorials, examples, documentation... separately from the [GitHub repo](https://github.com/AllenInstitute/bmtk).
- Installing from the source
  ```bash
  $ git clone https://github.com/AllenInstitute/bmtk.git
  $ cd bmtk
  $ python setup.py install
  ```
  - This method will  create a copy of the github repo on your computer, giving you access to all tutorials (in docs/tutorials/), examples (in examples/), documentation...
  - You will probably need to install python [dependencies](#dependencies) manually.
- Some of the tutorials or examples engines may require additional requirements to run.
- Additionally, if you want to take advantage of parallel computing on an HPC cluster, you also need the mpi4py python package.

## Dependencies

### NEURON

On Linux, you can treat NEURON like any other package and install it with `pip3/conda install`.
</br>On Windows, you need to download an installer.

| Pip (Linux)                    | Conda (Linux)                                 | Windows |
|--------------------------------|-----------------------------------------------|---------|
| ``` $ pip3 install neuron  ``` | ``` $ conda install -c conda-forge neuron ``` | [Click here for installaion guide](https://nrn.readthedocs.io/en/8.2.2/install/install_instructions.html#windows)|


### List of standard dependencies

- numpy
- h5py
- pandas
- matplotlib
- jsonschema
- (pytest: optional for running unit tests)
- (**mpi4py**: if you want to take advantage of parallel computing on an HPC cluster)


### [Installing dependencies](../background/packages.md)

| Pip                                               | Conda                                             |
|---------------------------------------------------|---------------------------------------------------|
| ``` $ pip3 install [package1] [package2] ...  ``` | ``` $ conda install [package1] [package2] ... ``` |
