# Copyright 2017. Allen Institute. All rights reserved
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
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import h5py
import csv
import pandas as pd

from bmtk.simulator.bionet.lifcell import LIFCell
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.stim import Stim
from bmtk.simulator.bionet.morphology import Morphology
from bmtk.simulator.bionet import nrn, io
from bmtk.simulator.bionet.property_schemas import CellTypes
import bmtk.simulator.bionet.config as cfg

from bmtk.simulator.bionet.biograph import BioEdge

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
        self._spike_trains_df = {}
        self._stim_networks = set()

        self._save_connections = False
        self._total_synapses = 0

        self._model_type_map = {
            'biophysical': BioCell,
            'point_process': LIFCell,
            'point_soma': lambda *x : None,
            'virtual': lambda *x : None
        }

        self._cell_model_gids = {model_type: [] for model_type in self._model_type_map.keys()}
        self._population_map = {}

        self._morphology_lookup = {}

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
    def save_connections(self):
        return self._save_connections

    @save_connections.setter
    def save_connections(self, value):
        # TODO: throw a warning if a user is trying to set save_connection = True after the network has been built
        self._save_connections = value

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
        #for node_pop in self.node_populations():
        #    for
        # TODO: Raise a warning if more than one internal population and no gids (node_id collision)
        # TODO: Verify there actually is at least one internal population
        for pop_name in self._graph._internal_pop_names:
            self._population_map[pop_name] = []  # TODO: Preallocate array
            for node, prop_map in self._graph.internal_nodes_itr(pop_name, rank, nhost):
                cell = self._model_type_map[node['model_type']](node, self, prop_map)
                self._cells[prop_map.gid(node)] = cell
                self._cell_model_gids[node['model_type']].append(prop_map.gid(node))
                # TODO: Need to make sure cell isn't virtual
                self._population_map[pop_name].append(cell)

        '''
        for node, prop_map in self._graph.internal_nodes_itr(rank, nhost):
            self._cells[prop_map.gid(node)] = self._model_type_map[node['model_type']](node, self, prop_map)
            self._cell_model_gids[node['model_type']].append(prop_map.gid(node))

            #print node.node_id, prop_map
        '''

        '''
        for node_pop in self._graph._internal_populations:
            for node in node_pop[rank::nhost]:
                print node.node_id
                print node['model_type']
                print self._model_type_map[node['model_type']](node, self)
                # exit()

        exit()
        '''


        # self._select_local_nodes()
        #for node in self._local_node_ids:

        '''
        for node in self._local_nodes:
            gid = node.node_id
            print

            if node.cell_type == CellTypes.Biophysical:
                self._cells[gid] = BioCell(node, self.spike_threshold, self.dL, self.calc_ecp, self._save_connections)
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
        '''
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
        for gid in self._cell_model_gids['biophysical']:
            cell = self._cells[gid]
            morphology_file = cell.morphology_file
            if morphology_file in self.__morphologies_cache:
                # create a single morphology object for each model_group which share that morphology
                morph = self.__morphologies_cache[morphology_file]

                # associate morphology with a cell
                cell.set_morphology(morph)
                self._morphology_lookup[morphology_file].append(gid)

            else:
                hobj = cell.hobj  # get hoc object (hobj) from the first cell with a new morphologys
                morph = Morphology(hobj)

                # associate morphology with a cell
                cell.set_morphology(morph)

                # create a single morphology object for each model_group which share that morphology
                self.__morphologies_cache[morphology_file] = morph
                self._morphology_lookup[morphology_file] = [gid]
        '''
        exit()
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

        '''
        io.log_info("Created morphologies")
        self._morphologies_built = True

    def set_seg_props(self):
        """Set morphological properties for biophysically (morphologically) detailed cells"""
        for _, morphology in self.__morphologies_cache.items():
            morphology.set_seg_props()

        io.log_info("Set segment properties")

    def calc_seg_coords(self):
        """Needed for the ECP calculations"""
        # TODO: Is there any reason this function can't be moved to make_morphologies()
        for morphology_file, morphology in self.__morphologies_cache.items():
            morph_seg_coords = morphology.calc_seg_coords()   # needed for ECP calculations

            for gid in self._morphology_lookup[morphology_file]:
                self._cells[gid].calc_seg_coords(morph_seg_coords)

        io.log_info("Set segment coordinates")

    def add_spikes_nwb(self, ext_net, nwb_file, trial):
        h5_file = h5py.File(nwb_file, 'r')
        self._spike_trains_ds[ext_net] = h5_file['processing'][trial]['spike_train']
        self._stim_networks.add(ext_net)

    def add_spikes_csv(self, ext_net, csv_file, sep=' '):
        spikes_df = pd.read_csv(csv_file, index_col=['gid'], sep=sep)
        self._spike_trains_df[ext_net] = spikes_df
        self._stim_networks.add(ext_net)

    def _get_spike_trains(self, src_gid, network):
        if network in self._spike_trains_ds:
            h5ds = self._spike_trains_ds[network]
            src_gid_str = str(src_gid)
            if src_gid_str in h5ds.keys():
                return h5ds[src_gid_str]['data']

        elif network in self._spike_trains_df:
            spikes_list = [float(t) for t in self._spike_trains_df[network].loc[src_gid]['spike-times'].split(',')]
            return spikes_list

        return []

    def make_stims(self):
        """Create the stims/virtual/external nodes.

        Make sure spike trains have been set before calling, otherwise it will creating spiking cells with no spikes.
        """
        #print self._stim_networks
        for pop_name in self._graph.virtual_pop_names():
            if pop_name not in self._stim_networks:
                continue

            self._stims[pop_name] = {}
            for node in self._graph.external_nodes_itr(pop_name, rank, nhost):
                # TODO: Put back
                spike_train = self._get_spike_trains(node.node_id, pop_name)
                self._stims[pop_name][node.node_id] = Stim(node, spike_train)

            self._init_connections()
            io.log_info('    Setting connections from {}'.format(pop_name))
            # TODO: skip if source_network is not in stims
            source_stims = self._stims[pop_name]
            syn_counter = 0
            #print self._population_map.keys()
            # TODO: Iterate and do node_id lookup at the same time, cache results (temp)
            for trg_pop in self._population_map.keys():
                for edge_pop in self._graph.external_edge_populations(src_pop=pop_name, trg_pop=trg_pop):
                    prop_maps = self._graph._edge_property_maps[edge_pop.name]
                    for trg_node in self._population_map[trg_pop]:
                        prop_map = prop_maps[trg_node.group_id]
                        for edge in edge_pop.get_target(trg_node.node_id):
                            bioedge = BioEdge(edge, self._graph, prop_map)
                            stim = source_stims[edge.source_node_id]
                            syn_counter += trg_node.set_syn_connection(bioedge, stim, stim)

        '''
            for trg_gid, trg_cell in self._cells.items():
                for trg_prop, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, source_network):
                    # TODO: reimplement weight function if needed
                    stim = source_stims[src_prop.node_id]
                    syn_counter += trg_cell.set_syn_connection(edge_prop, src_prop, stim)
            self._total_synapses += syn_counter
        '''

        '''
            for external_pop in self._graph._external_edges[pop_name]:
                print external_pop.target_population, "f"
                #for node in self._graph.external_nodes_itr(pop_name)

                # TODO: Keep track if virtual nodes are mixed, and if so check the source node
            #print pop_name
            # print virt_pop

        exit()
        for network in self._graph.external_networks():
            io.log_info('        %s cells' % network)

            if network not in self._stim_networks:
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
        '''

    def set_recurrent_connections(self):
        self._init_connections()
        syn_counter = 0
        for src_network in self._graph.internal_networks():
            io.log_info('    Setting connections from {}'.format(src_network))
            for trg_gid, trg_cell in self._cells.items():
                for trg_prop, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, src_network):
                    syn_counter += trg_cell.set_syn_connection(edge_prop, src_prop)
        self._total_synapses += syn_counter

    def set_external_connections(self, source_network):
        self._init_connections()
        io.log_info('    Setting connections from {}'.format(source_network))
        # TODO: skip if source_network is not in stims
        source_stims = self._stims[source_network]
        syn_counter = 0
        for trg_gid, trg_cell in self._cells.items():
            for trg_prop, src_prop, edge_prop in self._graph.edges_iterator(trg_gid, source_network):
                # TODO: reimplement weight function if needed
                stim = source_stims[src_prop.node_id]
                syn_counter += trg_cell.set_syn_connection(edge_prop, src_prop, stim)
        self._total_synapses += syn_counter

    def _init_connections(self):
        if not self._connections_initialized:
            io.log_info('Initializing connections.')
            for gid, cell in self._cells.items():
                cell.init_connections()
            self._connections_initialized = True

    def scale_weights(self, factor):
        io.log_info('Scaling all connection weights')
        for gid, cell in self.cells.items():
            cell.scale_weights(factor)

    def write_connections(self, output_dir, file_type='h5'):
        # TODO: this doesn't work on multi-node simulations
        assert(nhost == 1)

        # first write to a temp csv file
        tmp_csv_fname = os.path.join(output_dir, '.tmp_edges.csv')
        with open(tmp_csv_fname, 'w') as csvhandle:
            csvwriter = csv.writer(csvhandle, delimiter=' ')
            csvwriter.writerow(['trg_gid', 'src_gid', 'trg_network', 'src_network', 'segment', 'section', 'weight',
                                'delay', 'edge_type_id', 'connection_group'])
            for _, cell in self._cells.items():
                for conn in cell.get_connection_info():
                    csvwriter.writerow(conn)

        if file_type == 'csv':
            raise NotImplementedError()

        elif file_type == 'h5':
            # convert from csv to h5 format
            from bmtk.simulator.bionet.utils import edge_converter_csv
            edge_converter_csv(output_dir=output_dir, csv_file=tmp_csv_fname)

        # remove temp file
        os.remove(tmp_csv_fname)

    @classmethod
    def from_config(cls, config_file, graph):
        """A method for building a network from a config file.

        :param config_file: A json file (or object) with simulation parameters for loading NEURON network.
        :param graph: A BioGraph object that has already been loaded.
        :return: A BioNetwork object with nodes and connections that can be ran in a NEURON simulator.
        """
        io.log_info('Number of processors: {}'.format(nhost))
        io.log_info('Setting up network.')

        # 1. Load biophysical and point cells
        # 2. Load



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
        if 'spike_threshold' in run_dict:
            network.spike_threshold = run_dict['spike_threshold']
        if 'dL' in run_dict:
            network.dL = run_dict['dL']
        if 'calc_ecp' in run_dict:
            network.calc_ecp = run_dict['calc_ecp']

        # build the cells
        network.save_connections = config['output'].get('save_synapses', False)
        graph.io.log_info('Building cells.')
        #io.print2log('Building cells...')
        network.build_cells()

        # list of cells who parameters will be saved to h5
        if 'node_id_selections' in config and 'save_cell_vars' in config['node_id_selections']:
            network.save_gids(config['node_id_selections']['save_cell_vars'])
        # Find and save network stimulation. Do this before loading external/internal connections.

        if 'inputs' in config:
            for _, netinput in config['inputs'].items():
                if netinput['input_type'] == 'spikes' and netinput['module'] == 'nwb':
                    # Load external network spike trains from an NWB file.
                    # io.print2log0('Load input for {}'.format(netinput['network']))
                    network.add_spikes_nwb(netinput['node_set'], netinput['input_file'], netinput['trial'])

                elif netinput['type'] == 'external_spikes' and netinput['format'] == 'csv':
                    network.add_spikes_csv(netinput['source_nodes'], netinput['file'])

                # TODO: Allow for external spike trains from csv file or user function
                # TODO: Add Iclamp code.

            io.log_info('    Setting up external cells...')
            network.make_stims()

        io.log_info('Cells are built!')

        for netname in graph.external_networks():
            network.set_external_connections(netname)

        network.set_recurrent_connections()
        io.log_info('Network is built!')

        if network.save_connections:
            io.print2log0('Saving synaptic connections:')
            network.write_connections(config['output']['output_dir'])
            io.print2log0('    Synaptic connections saved to {}.'.format(config['output']['output_dir']))

        return network
