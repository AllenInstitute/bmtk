Tutorials
=========

.. toctree::
   :maxdepth: 2
   :titlesonly:

   Builder: Using the Network Builder <tutorial_NetworkBuilder_Intro>
   BioNet: Single cell with current injection <tutorial_single_cell_clamped>
   BioNet: Single with with synaptic input <tutorial_single_cell_syn>
   BioNet: Multiple Nodes with single cell-type <tutorial_single_pop>
   BioNet: Heterogeneous network <tutorial_multi_pop>
   PointNet: Point-neuron modeling <tutorial_pointnet_modeling>
   PopNet: Population-based firing rate models <tutorial_population_modeling>
   FilterNet: Full-field flashing movie <tutorial_filter_models>


Prerequisites
-------------

Running with Docker
+++++++++++++++++++

If you have Docker installed on your machine then there is a Docker Image with all the prerequists installed - including
a jupyter notebook server with the tutorials installed. Just run:

.. code:: bash

    $ docker pull alleninstitute/bmtk
    $ docker run -v /path/to/local/directory:/home/shared/workspace -p 8888:8888 alleninstitute/bmtk jupyter

and then open a browser to 127.0.0.1:8888/. The tutorials folder will contain the jupyter notebook tutorials for you to
follow along and modify. However, if you want to save the work permentately make sure to save it in the workspace
folder. The tutorials and examples folder will be deleted once the docker container has stopped.

you can also use the Docker image to run bmtk build and run scripts. Just replace the python <script>.py <opts> command
with docker run alleninstitute/bmtk -v /path/to/local/directory:/home/shared/workspace python <script>.py <opts>

Running from source
+++++++++++++++++++

The bmtk requires at minimum python 2.7 and 3.6+, as well as additional libraries to use features like building networks
or running analyses. To install the bmtk it is best recommending to pull the latest from github.

.. code:: bash

    $ git clone https://github.com/AllenInstitute/bmtk.git
    $ cd bmtk
    $ python setup.py install

However, to run a simulation on the network the bmtk uses existing open-source simulatiors, which (at the moment) needs
to be installed separately. The different simulators, which run simulations on different levels-of-resolution, will
require different software. So depending on the type of simulation to be run

biophysically detailed network (BioNet) - Uses NEURON.
point-neuron network (PointNet) - Uses NEST.
population-level network (PopNet) - Uses DiPDE.
filter models of the visual field (FilterNet) - Uses LGNModels