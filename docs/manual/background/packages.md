[Back to manual](/docs/manual/README.md)

# Python package management 

In general, it is best to create a virtual environment before installing the python packages you need for a certain project. This way, the package versions required for compatibility with one project don't interfere with the compatibility of another project. You will only need to create the environment and install the packages once. In subsequent sessions, you just need to activate the environment you previously made.

## Setting up a virtual environment

| Command                     | Pip                                         | Conda                           |
|-----------------------------|---------------------------------------------|---------------------------------|
| Creating an environment     | ``` $ python3 -m venv [path/to/env] ```     | ``` $ conda create -n [env] ``` |
| Activating an environment   | ``` $ source [path/to/env]/bin/activate ``` | ``` $ conda activate [env] ```  |
| Deactivating an environment | ``` $ deactivate ```                        | ``` $ conda deactivate ```      |

## Managing packages

| Command                    | Pip                                               | Conda                                             |
|----------------------------|---------------------------------------------------|---------------------------------------------------|
| Installing packages        | ``` $ pip3 install [package1] [package2] ...  ``` | ``` $ conda install [package1] [package2] ... ``` |
| List of installed packages | ``` $ pip3 list ```                               | ``` $ conda list ```                              |

## [Other useful terminal commands](./terminal.md)