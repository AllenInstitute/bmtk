Simulating example networks with BioNet
=======================================

To get started with running simulations with BioNet, we recommend to run network examples provided in the directory docs/examples/simulators/bionet:

 * /14cells : network of 14 cell receiving an external input.
 * /450cells : network of 450 cells receiving an external input

Inside each example folder you will find:

Model description
-----------------

`network` : folder including files describing parameters of the cells and their connections

Cells in modeled networks may be generally categorized as: 1) the simulated cells for which dynamics is explicitly simulated using NEURON and 2) external input cells for which spiking dynamics is not simulate explicitly, but instead loaded from files. The exernal input cell can influence the simulate cells but not vice versa and thus are convenient to use in order to simulate feedforward input.

The example networks includes the simulated cells from primary mouse visual cortex (V1) receiving external inputs from the Lateral Geniculate Nucleu (LGN) and also background input in a form of a Travelling Wave (TW) as shown in the Figure below:

.. image:: _static/images/all_network_cropped.png
   :scale: 15 %

The figure depicts cells in the 14 cell network where among the 14 simulated cells there are 10 biophysically detailed cells and 4 LIF cells.

Network may be viewed as a graph with nodes of a graph representing cells and edges of a graph representing connections. The parameters of the network, thus can be specified by describing parameters of nodes and edges.
In our reprepsentation of a graph, nodes (edges) belong to a particular node (edge) type, thus possessing properties of that type, but also possesing properties specific to that individual node (edge). Thus, properties of nodes are described using two files: node_file, node_type_file (edge_file, edge_type_file). Similarly  edges are described using two files (edges_file, edge_types_file). For node_type_file and edge_type_file we use csv format whereas for node_file and edge_file used HDF5 format.


**Nodes**:

Simulated cells

 * v1_nodes.h5 : V1 nodes
 * v1_node_types.csv : V1 node types

External cells:

LGN cells

 * lgn_nodes.h5 : LGN nodes
 * lgn_node_types.csv : LGN node types

TW cells

 * tw_nodes.h5 TW nodes
 * tw_node_types.csv TW node types


**Edges**:

 * v1_v1_edges.h5 : V1 => V1 edges
 * v1_v1_edge_types.csv : V1 => V1 edge types

 * lgn_v1_edges.h5 : LGN => V1 edges
 * lgn_v1_edge_types.csv LGN => V1 edge types

 * tw_v1_edges.h5 : TW => V1 edge
 * tw_v1_edge_types.csv TW => V1 edge types

The spikes times of external cells are precomuted and provided in the directory docs/examples/simulator/NWB_files.

Each network utilize models of individual cells, synapse and recording electrodes defined in docs/examples/simulators/bionet/components/mechanisms directory.

The paths to each of these files and directories are specified in the configuration file:


Configuring simulation
-----------------------

All the files listed above and describing the network are listed in the configuration file `config.json` to instruct BioNet where to look for files describing network. Additionally, the configuration file includes simulation parameters (e.g. duration of simulation and time step, etc).

Please refer to the `configuration file documentation <./bionet_config.html>`_ for details.


Running simulation
-----------------

Running simualtions requires the following Python scripts

`run_bionet.py` : main python script which calls BioNet to run simulation

`set_cell_params.py`: module setting properties of cells 
`set_syn_params.py`: module setting properties of synapses
`set_weights.py` : modules allowign to set parameter dependent connection weights

The example network use biophysically-detailed models of individual cells, and require additional NEURON channel mechanisms describing dynamics of ionic channels. To compile these NEURON mechanims go to the subdirectory docs/examples/simulators/bionet/components/mechanisms and run the NEURON command:
::
   nrnivmodl modfiles/

From the directory of the network example you can run a simulation on a single core by executing the main python script with configuration file as a comman line argument as follows:
::
  python run_bionet.py config.json

Or to run in parallel with MPI on $NCORES CPUs:
::
  mpirun -np $NCORES nrniv -mpi -python run_bionet config.json

In either case, the main script will load the configuration file containing paths to files describing the network and will load and simulate the network. 

When simulation is completed you will see a message “Simulation completed”.

BioNet allows saving simulation output in blocks while simulation is still running, giving users an ability to check and analyze intermediate output. During the run you will see some output reporting on the progress of a simulation as follows:


When simulation completed you will see a message "Simulation completed".

Simulation output
-----------------

The output directory includes:
 * spikes.h5 : HDF5 file containg the spikes of the simulated cells.
 * cellvars/N.h5 : HDF5 file containing time series recordings of somatic variables  (e.g., somatic voltage, [Ca++]) for cell with node_id=N (there might be multiple such files, up to the number of cells in the model, or none at all, depending on the settings in the simulation config).
 * config.json : a copy of configuration for record keeping
 * log.txt : run log file including time-stamped information about the progress of a simulation.


Upon completion you may run the script plot_rasters.py to plot spike rasters of external as well as simulated cells:
::
  python plot_rasters.py

**External input cells**

.. image:: _static/images/ext_inputs_raster.png
   :scale: 20 %

**Simulated cells**

.. image:: _static/images/v1_raster.png
   :scale: 20 %



Simulating your network models
------------------------------

To run simulations of your network with BioNet, you will first need to provide a pre-built network in the format understood by BioNet. We recommend using `BMTK's network builder api <builder>`_, but you may also use your own scripts or a third party tool to build a network. As a start we suggest to modify the existing network examples as a quick way of customizing network models and then build your own model following `builders examples tutorial <https://github.com/AllenInstitute/bmtk/tree/develop/docs/examples/builder/bionet_14cells>`_.

When you have your custom model built, you will need to specify in the your configuration file the paths to the network, components as well as simulation run parameters.

Just as in the above examples, your run folder should include Python modules: set_cell_params.py, set_syn_params.py, set_weights.py specifying how models of cells, synapses and connection weights are created as well as a main python script. 

When running different simulations you will rarely need to modify the main Python script running BioNet. Instead, you will commonly need to modify paths to network files or run parameters in the configuration file  to instruct BioNet which model to run and how to run it. Please refer to the `configuration file documentation <./bionet_config.html>`_ for details.


