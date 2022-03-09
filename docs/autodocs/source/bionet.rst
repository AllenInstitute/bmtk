BioNet
======

.. figure:: _static/images/bmtk_architecture_bionet_highlight.jpg
   :scale: 40%

BioNet is a high-level interface to `NEURON <http://neuron.yale.edu/neuron/>`_ that facilitates simulations of
large-scale networks of multicompartmental neurons. Some of its main features include:

* Automatically integrates MPI for parallel simulations without the need of extra coding.

* Supports models and morphologies from the Allen `Cell-Types Database <http://celltypes.brain-map.org/data>`_, as well
  as custom hoc and NeuroML2 cell and synapse models.

* Use spike-trains, synaptic connections, current clamps or even extracellular stimulation to drive network firing.

* Can simulate extracellular field recordings.


Inputs
--------
Inputs can be specified in the “inputs” sections of the `simulation config <./simulators.html#configuration-files>`_,
following the rules specified in the `SONATA Data format <https://github.com/AllenInstitute/sonata>`_.

Spike-Trains
++++++++++++
The modeler may wish to have certain cells in the circuit generate a pre-arranged series of spikes to drive the network.
These cells must have ``model_type`` value ``virtual`` and are not actual cell objects (you can’t record from them). You
may use either a `SONATA spike file <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#spike-file>`_,
an NWB file, or a space-separated csv file with columns **node_id**, **population**, and **timestamps**. The following
shows some examples of how to generate `spike-train files using bmtk <./analyzer.html#creating-spike-trains>`_.

.. code:: json

    {
        "LGN_spikes": {
            "input_type": "spikes",
            "module": "sonata",
            "input_file": "./inputs/lgn_spikes.h5",
            "node_set": {"population": "lgn"}
        }
    }

* module:  either sonata, hdf5, csv, or nwb: depending on the format of the spikes file

* `node_set <./simulators.html#node-sets>`_: used to filter which cells will receive the inputs

* input_file: path to file contain spike-trains for one or mode node



Current Clamp
+++++++++++++
May use one step current clamp on multiple nodes, or have one node receive multiple current injections. Currently ;)
only support injections at the soma.

.. code:: json

    {
        "current_clamp_1": {
            "input_type": "current_clamp",
            "module": "IClamp",
            "node_set": "biophys_cells",
            "amp": 0.1500,
            "delay": 500.0,
            "duration": 500.0
        }
    }

* module:  Always IClamp
* `node_set <./simulators.html#node-sets>`_: used to filter which cells will receive the inputs
* amp: injection in pA
* delay: onset of current injection in ms
* duration: duration of current injection in ms

Voltage Clamp
+++++++++++++

Extracellular Stimulation
+++++++++++++++++++++++++
Allows for a set of external electrodes to provide a continuous stimulation in the neuropil. Requires a space-separated csv file with one row for each electrode:

.. code::
    :name: xstim_electrode.csv

    ip pos_x pos_y pos_z rotation_x rotation_y rotation_z
    0 6.1803398874989481 0.0 19.021130325903069 0.0 0.0 0.0

And in the configuration file

.. code:: json

    {
        "extra_stim": {
            "input_type": "lfp",
            "module": "xstim",
            "node_set": "all",
            "positions_file": "./inputs/xstim_electrode.csv",
            "waveform": {
                "shape": "sin",
                "del": 1000.0,
                "amp": 0.100,
                "dur": 2000.0,
                "freq": 8.0
            }
        }
    }

* module:  Always xstim
* `node_set <./simulators.html#node-sets>`_: used to filter which cells will receive the inputs
* positions_file: onset of current injection in ms
* waveform: form on the input, requires arguments “shape”, “amp” (in pA), “del” (delay in ms) and “dur” (duration in ms). Shape may either be “dc” or “sin” (with optional arguments “freq”, “phase” and “offset”)


Spontaneous Firing
++++++++++++++++++


Outputs
-------
Spikes
++++++
By default all non-virtual cells in the circuit will have all their spikes at the soma recorded. The “spike_threadhold”
parameter in the “run” block of the simulation config is used to determine what counts as a spike for a conductance model
cell.


Membrane and Intracellular Variables
++++++++++++++++++++++++++++++++++++
Used to record the time trace of specific cell variables, usually the membrane potential (v). For multi-compartmental
cells the report can record from any segment that contains mechanics for the desired variable. See
`SONATA docs <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#simulation-output---reports>`_
for more information about how multi-segment recordings are represented.

.. code:: json

    {
        "calcium_concentration": {
            "module": "membrane_report",
            "cells": {"population": "biophysical"},
            "variable_name": "cai",
            "sections": "soma",
            "file_name": "cai_traces.h5"
        },
        "membrane_potential": {
            "module": "membrane_report",
            "cells": {"node_ids": [0, 1, 2, 3, 4, 5]},
            "variable_name": "v",
            "sections": "all",
            "file_name": "cai_traces.h5"
        }
    }

* variable_name: name of variable being recorded, will depend on the cell model
* cells: a `node_set <./simulators.html#node-sets>`_ to filter out what cells to record.
* sections: either “all”, “soma”, “basal” or “apical”
* file_name: name of file where traces will be recorded, under the “output_dir”. If not specified the the report title
  will be used, eg “calcium_concentration.h5” and “membrane_potential.h5”

.. warning::
	Disk space can be an issue when recording membrane variables. For large networks recording all segments or all cells, every for a 1 second simulation, can cause BMTK to try to write output files in the 100’s of GB or even TB.



Extracellular Potential
+++++++++++++++++++++++
Will simulate recording from an extracellular electrode placed in the neuropil. See
`SONATA documentation <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#extracellular-report>`_.
Requires a space-separated csv file to specify the location of each recording channel:

.. code::
    :name: ./components/xelectrode/linear_probe.csv

    channel x_pos y_pos z_pos
    0 10.0 0 5.0
    1 10.0 -10 5.0
    2 10.0 -20 5.0
    3 10.0 -30 5.0
    4 10.0 -40 5.0

And in the config

.. code:: json

    {
        "ecp": {
            "cells": "all",
            "variable_name": "v",
            "module": "extracellular",
            "electrode_positions": "components/xelectrode/linear_probe.csv",
            "file_name": "ecp.h5",
            "contributions_dir": "ecp_contributions"
        }
    }

* cells: a `node_set <./simulators.html#node-sets>`_ to filter out what cells will contribute to the ecp.
* variable_name: name of contributing variable, v for membrane potential
* electrode_positions: name of electrode placement file
* contributions_dir: The output ecp file will contain the combined contributes from all cells and not possible to
  determine the ecp of each individual cell. But if “contributions_dir” is specified it will also record and save each
  individual cells’ ecp.


Synaptic Variables
++++++++++++++++++
Similar to recording from membrane potential, by setting ``module`` parameter to ``netcon_report`` you can record the
variables from a synapse. The output is similar to a
`SONATA membrane report <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#frame-oriented-node-element-recordings>`_,
but instead of each column being a segment of a neuron, each column represents a different synapses.

.. code:: json

    {
        "syn_report": {
            "cells": {"model_type": "biophysical"},
            "variable_name": "tau1",
            "module": "netcon_report",
            "sections": "soma",
            "syn_type": "Exp2Syn"
        }
    }



Advanced Options
----------------

Specifying Synapse locations
++++++++++++++++++++++++++++
In SONATA the location of each synapse is determined by the
`"afferent_section_id" and "afferent_section_pos"attributes <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#edges---required-attributes>`_,
which requires modelers to know how NEURON parses the morphology of each cell. If these parameters are specified in the edges file
SONATA will use them to place a synapse on the target.

Alternatively BMTK supports the option parameters “distance_range" and “target_sections”, which if present in the edges
file, will direct BMTK to randomly choose a target synapse location under the limitations. Here “target_setions” refers
to a neuronal area (somatic, axon, apical, basal) and “"istance_range” is the minimum and maximum arc-length distance
(in um) from the soma to place the synapse. For example to specify synapses be created either at the soma or nearby
basal dendrites:


.. code::
    :name: edge_type.csv

    edge_type_id distance_range target_sections ...
    100 "[0.0, 100.0]" "['somatic', 'basal']" ...

Using parameters “distance_range” and “target_sections” will speed up the instantiation by a bit. And has a benefit
that the modeler doesn’t need to know the full details of the target_morphology. It may cause results to vary, but in
our experience for large-networks usually doesn’t change the dynamics.



