# PointNet 450 GLIF neuron network

This is a small example network that uses the Allen Institute glif cell models. The network is recurrent and receives
stimulation from external spike trains (virtual cells).



## Network files
The *network* directory contains [SONATA](https://github.com/AllenInstitute/sonata) formated files describing the network. These
files were built using the *build_network.py* script. To rebuild the network files install the BMTK and run
```bash
$ python build_network.py
```

This will overwrite the existing files in the network directory. Also note that the script uses some randomness to
determine the set synaptic connections, so expect different simulation results everytime you rebuild the network.

### The GLIF neuron Models

The network uses models and fitted data taken from [The Allen Cell Types Database](https://celltypes.brain-map.org/data). The
**model_template** attribute specifies the glif model version (lif, lif_r, lif_asc, lif_r_asc, or glif_r_asc_a). The
**dynamics_params** attribute points to a json file in *../point_components/cell_models*. These json files were downloaded
from the Cell Types database from selected mouse primary visual area cells.


## Running the Simulation

### Requirements:

To run the simulation, you must first have NEST 2.11 installed with python bindings. You must also have the glif NEST
modules installed to use the **nest:glif** models. To do so, [follow the instructions on the GLIF2NEST repo](https://github.com/AllenInstitute/GLIF2NEST/).

### Simulation parameters

The file *config.json* contains all the parameters required to run the full simulation. Including run-time parmaters ("run" section),
the input file used to generated incoming spike trains ("inputs" section), and the outputs ("output" and "reports" section).

To run a simulation
```
$ python run_pointnet.py config.simulation.json
```

If you have access to a HPC with MPI libraries installed:
```
$ mpirun -n <N> python run_point.py config.simulation.json
```

When it's finsihed an *output/* directory that contains spikes (and other variables if specified in "reports" section). You
can use the *plot_spikes.py* script to quickly create a raster plot.


