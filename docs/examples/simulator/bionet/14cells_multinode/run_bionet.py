# -*- coding: utf-8 -*-

"""Simulates an example network of 14 cell receiving two kinds of exernal input as defined in configuration file"""


import sys, os
import h5py
import numpy as np
from numpy.random import randint
from neuron import h

import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork

from bmtk.utils.io import TabularNetwork_AI
from bmtk.simulator.bionet.property_schemas import AIPropertySchema


import set_weights
import set_cell_params
import set_syn_params

pc = h.ParallelContext()
MPI_RANK = int(pc.id())


def check_cellvars(gid, conf):
    expected_file = 'expected/cellvars/{}.h5'.format(gid)
    results_file = os.path.join(conf['output']['cell_vars_dir'], '{}.h5'.format(gid))
    assert (os.path.exists(results_file))

    results_h5 = h5py.File(results_file, 'r')
    expected_h5 = h5py.File(expected_file, 'r')
    for variable in conf['run']['save_cell_vars']:
        assert (variable in results_h5)
        results_data = results_h5[variable].value
        expected_data = expected_h5[variable].value
        # consider using np.allclose() since the percision can vary depending on neuron version
        assert (np.array_equal(results_data, expected_data))

def check_ecp():
    SAMPLE_SIZE = 100
    expected_h5 = h5py.File('expected/ecp.h5', 'r')
    nrows, ncols = expected_h5['ecp'].shape
    expected_mat = np.matrix(expected_h5['ecp'])
    results_h5 = h5py.File('output/ecp.h5', 'r')
    assert ('ecp' in results_h5.keys())
    results_mat = np.matrix(results_h5['ecp'][:])

    assert (results_h5['ecp'].shape == (nrows, ncols))
    for i, j in zip(randint(0, nrows, size=SAMPLE_SIZE), randint(0, ncols, size=SAMPLE_SIZE)):
        assert (results_mat[i, j] == expected_mat[i, j])


def run(config_file):
    conf = config.from_json(config_file)        # build configuration
    io.setup_output_dir(conf)                   # set up output directories
    nrn.load_neuron_modules(conf)               # load NEURON modules and mechanisms
    nrn.load_py_modules(cell_models=set_cell_params,  # load custom Python modules
                        syn_models=set_syn_params,
                        syn_weights=set_weights)

    graph = BioGraph.from_config(conf,  # create network graph containing parameters of the model
                                 network_format=TabularNetwork_AI,
                                 property_schema=AIPropertySchema)

    net = BioNetwork.from_config(conf, graph)   # create netwosim = Simulation.from_config(conf, network=net)  rk of in NEURON
    sim = Simulation.from_config(conf, network=net)         # initialize a simulation
    # sim.set_recordings()                        # set recordings of relevant variables to be saved as an ouput
    sim.run()                                   # run simulation

    if MPI_RANK == 0:
        # Check spikes
        assert (os.path.exists(conf['output']['spikes_ascii_file']))
        assert (spike_files_equal(conf['output']['spikes_ascii_file'], 'expected/spikes.txt'))

        # Check extracellular recordings
        check_ecp()

        # Check saved variables
        for saved_gids in conf['node_id_selections']['save_cell_vars']:
            check_cellvars(saved_gids, conf)

    """
    "cell_vars_dir": "$OUTPUT_DIR/cellvars",

    "save_cell_vars": ["v", "cai"],

    "node_id_selections": {
        "save_cell_vars":
    """

    '''
    # Test the results of the ecp
    SAMPLE_SIZE = 100
    expected_h5 = h5py.File('expected/ecp.h5', 'r')
    nrows, ncols = expected_h5['ecp'].shape
    expected_mat = np.matrix(expected_h5['ecp'])
    results_h5 = h5py.File('output/ecp.h5', 'r')
    assert ('ecp' in results_h5.keys())
    results_mat = np.matrix(results_h5['ecp'][:])

    assert (results_h5['ecp'].shape == (nrows, ncols))
    for i, j in zip(randint(0, nrows, size=SAMPLE_SIZE), randint(0, ncols, size=SAMPLE_SIZE)):
        assert (results_mat[i, j] == expected_mat[i, j])


    for gid in range(0, 11):
        assert(os.path.exists('output/cellvars/{}.h5'.format(gid)))
    '''

    pc.barrier()
    nrn.quit_execution()                        # exit


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
