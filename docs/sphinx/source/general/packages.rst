.. _packages:

Python package management
=========================

In general, it is best to create a virtual environment before installing
the python packages you need for a certain project. This way, the
package versions required for compatibility with one project don’t
interfere with the compatibility of another project. You will only need
to create the environment and install the packages once. In subsequent
sessions, you just need to activate the environment you previously made.

Setting up a virtual environment
--------------------------------

+------------------+-----------------------------------------+----------------------------+
| Command          | Pip                                     | Conda                      |
+==================+=========================================+============================+
| Creating an      | ``$ python3 -m venv [path/to/env]``     | ``$ conda create -n [env]``|
| environment      |                                         |                            |                      
+------------------+-----------------------------------------+----------------------------+
| Activating an    | ``$ source [path/to/env]/bin/activate`` | ``$ conda activate [env]`` |
| environment      |                                         |                            |
+------------------+-----------------------------------------+----------------------------+
| Deactivating an  | ``$ deactivate``                        | ``$ conda deactivate``     |
| environment      |                                         |                            |
+------------------+-----------------------------------------+----------------------------+

Managing packages
-----------------

+--------------+---------------------------+---------------------------+
| Command      | Pip                       | Conda                     |
+==============+===========================+===========================+
| Installing   | ``$ pip3 install          | ``$ conda install         |
| packages     | [package1] [package2]     | [package1] [package2]     |
|              | ...``                     | ...``                     |
+--------------+---------------------------+---------------------------+
| List of      | ``$ pip3 list``           | ``$ conda list``          |
| installed    |                           |                           |
| packages     |                           |                           |
+--------------+---------------------------+---------------------------+

:ref:`Other useful terminal commands <terminal>`
------------------------------------------------
