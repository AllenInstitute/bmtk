# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the Allen Institute's written permission. For
# purposes of this license, commercial purposes is the incorporation of the Allen Institute's software into anything for
# which you will charge fees or other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import h5py

from bmtk.simulator.bionet.lifcell import LIFCell
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.stim import Stim
from bmtk.simulator.bionet.morphology import Morphology
from bmtk.simulator.bionet import nrn, io
from bmtk.simulator.bionet.property_schemas import CellTypes
import bmtk.simulator.bionet.config as cfg

# TODO: leave this import, it will initialize some of the default functions for building neurons/synapses/weights.
import bmtk.simulator.bionet.default_setters

from neuron import h

pc = h.ParallelContext()  # object to access MPI methods
nhost = int(pc.nhost())
rank = int(pc.id())


class BioNetwork(object):
    """Class for instantiating a NEURON network will cell and synaptic hoc objects.

    Takes a BioGraph class with nodes and edges, and converts it to a network with NEURON Cells and Synapses. When
    possible use the factory methods (from_json, ...) to automatically build a network from an existing setup.
    """

    def __init__(self, graph):
        """

        :param graph: BioGraph object
        """
        self.__spike_threshold = -15.0  # membrane voltage of spike for a biophysical cell
        self.__dL = 20  # max length of a morphology segement
        self.__calc_ecp = False  # for calculating extracellular field potential
        self._cells_built = False
        self._morphologies_built = False
        self._connections_initialized = False

        self._graph = graph

        self._local_nodes = []  # All nodes objects that exists on this rank
        self._local_node_gids = []  # All gids on this rank
        self._local_node_types = {}
        self._local_biophys_gids = []
        self._local_lif_gids = []
        self._saved_gids = []  # GIDs specified in "groups", used for saving membrane-potential, Ca++ flux, etc
        self._cells = {}  # table of Cell-Type objects searchable by gid

        self.__morphologies_cache = {}  # Table of saved morphology files
        self._stims = {}  # dictionary of external/stim/virtual nodes by [network_name][gid]
        self._spike_trains_ds = {}  # save nwb spike-train datasets for when stims need to be built

    @property
    def spike_threshold(self):
        return self.__spike_threshold

    @spike_threshold.setter
    def spike_threshold(self, potential):
        self.__spike_threshold = potential

    @property
    def dL(self):
        return self.__dL

    @dL.setter
    def dL(self, segment_length):
        self.__dL = segment_length

    @property
    def calc_ecp(self):
        return self.__calc_ecp

    @calc_ecp.setter
    def calc_ecp(self, value):
        self.__calc_ecp = value

    @property
    def gids(self):
        return self._local_node_gids

    @property
    def biopyhys_gids(self):
        return self._local_biophys_gids

    @property
    def saved_gids(self):
        return self._saved_gids

    @property
    def cells(self):
        return self._cells

    def build_cells(self):
        """Instantiate cells based on parameters provided in the InternalCell table and Internal CellModel table"""
        self._select_local_nodes()
        for node in self._local_nodes:
            gid = node.node_id

            if node.cell_type == CellTypes.Biophysical:
                self._cells[gid] = BioCell(node, self.spike_threshold, self.dL)
                self._local_biophys_gids.append(gid)

            elif node.cell_type == CellTypes.Point:
                self._cells[gid] = LIFCell(node)

            elif node.cell_type == CellTypes.Virtual:
                # Just in case, should never see
                continue

            else:
                io.print2log0('ERROR: not implemented class')
                # raise NotImplementedError('not implemented cell class')
                nrn.quit_execution()

            # TODO: Add ability to easily extend the Cell-Types without hardcoding into this loop!!
        pc.barrier()  # wait for all hosts to get to this point

        self.make_morphologies()
        self.set_seg_props()  # set segment properties by creating Morphologies
        # self.set_tar_segs()  # set target segments needed for computing the synaptic innervations
        self.calc_seg_coords()  # use for computing the ECP
        self._cells_built = True

    def save_gids(self, gid_list):
        """List of cell GIDs whose variables (besides spikes) will be saved to h5.

        :param gid_list: List of existing internal gids in network.
        """
        # TODO: check that gid's in list exists
        saved_gids_set = set(gid_list)
        local_gids_set = set(self._local_node_gids)
        self._saved_gids = list(saved_gids_set & local_gids_set)

    def _select_local_nodes(self):
        """Divide all possible nodes among the various ranks (machines) for MPI usage. For single-processor simulation
        all nodes will be local."""
        all_nodes = self._graph.get_internal_nodes()
        for node in all_nodes[rank::nhost]:
            # Simple round-robin spliting of nodes. i.e. Machine i of N will have nodes i, i+N, i+2N, etc.
            self._local_nodes.append(node)
            self._local_node_gids.append(node.node_id)

            # saves node by node_type_id. Is this used anymore?
            if node['node_type_id'] in self._local_node_types:
                self._local_node_types[node.node_type_id].append(node)
            else:
                self._local_node_types[node.node_type_id] = [node]

    def make_morphologies(self):
        """Creating a Morphology object for each biophysical model"""
        for node in self._local_nodes:
            if node.cell_type == CellTypes.Biophysical:
                node_type_id = node.node_type_id
                morphology_file = node.morphology_file
                if node_type_id in self.__morphologies_cache:
                    # create a single morphology object for each model_group which share that morphology
                    morph = self.__morphologies_cache[node_type_id]

                    # associate morphology with a cell
                    self._cells[node.node_id].set_morphology(morph)

                else:
                    hobj = self._cells[node.node_id].hobj  # get hoc object (hobj) from the first cell with a new morphologys
                    morph = Morphology(hobj)

                    # associate morphology with a cell
                    self._cells[node.node_id].set_morphology(morph)

                    # create a single morphology object for each model_group which share that morphology
                    self.__morphologies_cache[node_type_id] = morph

        io.print2log0("Created morphologies")
        self._morphologies_built = True

    def set_seg_props(self):
        """Set morphological properties for biophysically (morphologically) detailed cells"""
        for _, morphology in self.__morphologies_cache.items():
            morphology.set_seg_props()

        io.print2log0("Set segment properties")

    def calc_seg_coords(self):
        """Needed for the ECP calculations"""
        for node_type_id, morphology in self.__morphologies_cache.items():
            morph_seg_coords = morphology.calc_seg_coords()   # needed for ECP calculations

            for node in self._local_node_types[node_type_id]:
                self._cells[node.node_id].calc_seg_coords(morph_seg_coords)

        io.print2log0("Set segment coordinates")

    def add_spikes_nwb(self, network, nwb_file, trial):
        h5_file = h5py.File(nwb_file, 'r')
        self._spike_trains_ds[network] = h5_file['processing'][trial]['spike_train']

    def _get_spike_trains(self, src_gid, network):
        if network in self._spike_trains_ds:
            h5ds = self._spike_trains_ds[network]
            src_gid_str = str(src_gid)
            if src_gid_str in h5ds.keys():
                return h5ds[src_gid_str]['data']

        return []

    def make_stims(self):
        """Create the stims/virtual/external nodes.

        Make sure spike trains have been set before calling, otherwise it will creating spiking cells with no spikes.
        """
        for network in self._graph.external_networks():
            if network not in self._spike_trains_ds:
                continue

            self._stims[network] = {}
            # Get a list of the external gid's that connect to cells on this node.
            src_gids_set = set()
            for trg_gid, trg_cell in self._cells.items():
                # TODO: Create a function that can return a list of all src_gids
                for trg_prop, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, network):
                    src_gids_set.add(src_prop.node_id)  # TODO: just store the src_prop

            # Get the spike trains of each external node and create a Stim object
            for src_gid in src_gids_set:
                src_prop = self._graph.get_node(src_gid, network)
                spike_train = self._get_spike_trains(src_gid, network)
                self._stims[network][src_gid] = Stim(src_prop, spike_train)

    def set_recurrent_connections(self):
        self._init_connections()
        io.print2log0('Setting internal connections...')
        syn_counter = 0
        for src_network in self._graph.internal_networks():
            for trg_gid, trg_cell in self._cells.items():
                for trg_prop, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, src_network):
                    syn_counter += trg_cell.set_syn_connection(edge_prop, src_prop)

        io.print2log0('    set %d synapses' % syn_counter)

    def set_external_connections(self, source_network):
        self._init_connections()
        io.print2log0('Setting connections from  %s' % source_network)
        source_stims = self._stims[source_network]
        syn_counter = 0
        for trg_gid, trg_cell in self._cells.items():
            for trg_prop, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, source_network):
                # TODO: reimplement weight function if needed
                stim = source_stims[src_prop.node_id]
                syn_counter += trg_cell.set_syn_connection(edge_prop, src_prop, stim)

        io.print2log0('    set %d synapses' % syn_counter)

    """
    def set_external_connections(self):
        self._init_connections()
        for network in self._graph.external_networks():
            # TODO: don't do this?!?!
            if network not in self._stims:
                continue

            io.print2log0('Setting connections from  %s' % network)
            src_network = self._stims[network]
            syn_counter = 0
            for trg_gid, trg_cell in self._cells.items():
                for trg_prop, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, network):
                    # TODO: reimplement weight function if needed
                    stim = src_network[src_prop.node_id]
                    trg_cell.set_syn_connection(edge_prop, src_prop, stim)
                    syn_counter += 1

            io.print2log0('    set %d synapses' % syn_counter)
    """

    def _init_connections(self):
        if not self._connections_initialized:
            io.print2log0('Initializing all connections. Will flush all if exist')
            for gid, cell in self._cells.items():
                cell.init_connections()
            self._connections_initialized = True

    def scale_weights(self, factor):
        io.print2log0('Scaling all connection weights')
        for gid, cell in self.cells.items():
            cell.scale_weights(factor)

    @classmethod
    def from_config(cls, config_file, graph):
        """A method for building a network from a config file.

        :param config_file: A json file (or object) with simulation parameters for loading NEURON network.
        :param graph: A BioGraph object that has already been loaded.
        :return: A BioNetwork object with nodes and connections that can be ran in a NEURON simulator.
        """
        io.print2log0('Setting up network...')

        # load the json file or object
        if isinstance(config_file, basestring):
            config = cfg.from_json(config_file, validate=True)
        elif isinstance(config_file, dict):
            config = config_file
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(config_file, type(config_file)))
        network = cls(graph)

        if 'run' not in config:
            raise Exception('Json file is missing "run" entry. Unable to build Bionetwork.')
        run_dict = config['run']

        # Overwrite default network parameters if they exists in the config file
        io.print2log0('Loading network parameters.')
        if 'spike_threshold' in run_dict:
            network.spike_threshold = run_dict['spike_threshold']
        if 'dL' in run_dict:
            network.dL = run_dict['dL']
        if 'calc_ecp' in run_dict:
            network.calc_ecp = run_dict['calc_ecp']

        # build the cells
        io.print2log('Building cells')
        network.build_cells()
        io.print2log0('Cells are instantiated')

        # list of cells who parameters will be saved to h5
        if 'groups' in config and 'save_vars' in config['groups']:
            network.save_gids(config['groups']['save_vars'])

        # Find and save network stimulation. Do this before loading external/internal connections.
        if 'input' in config:
            for netinput in config['input']:
                if netinput['type'] == 'external_spikes' and netinput['format'] == 'nwb' and netinput['active']:
                    # Load external network spike trains from an NWB file.
                    io.print2log0('Add stims for {}'.format(netinput['network']))
                    network.add_spikes_nwb(netinput['network'], netinput['file'], netinput['trial'])
                # TODO: Allow for external spike trains from csv file or user function
                # TODO: Add Iclamp code.

            io.print2log0('Setting up stims')
            network.make_stims()

        # load internal and external connections
        # for external networks, set to default true and turn off only if explicitly set in the config
        external_network_settings = {name: True for name in graph.external_networks()}
        if 'connect_external' in run_dict:
            external_network_settings.update(run_dict['connect_external'])
        for netname, connect in external_network_settings.items():
            if connect:
                network.set_external_connections(netname)
                # network.set_external_connections()

        if run_dict['connect_internal']:
            network.set_recurrent_connections()

        return network
