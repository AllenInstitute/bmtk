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
import numpy as np
from neuron import h

from bmtk.simulator.core.simulator_network import SimNetwork
from bmtk.simulator.core import sonata_reader
from bmtk.simulator.bionet.biocell import BioCell, BioCellSpontSyn
from bmtk.simulator.bionet.pointprocesscell import PointProcessCell, PointProcessCellSpontSyns
from bmtk.simulator.bionet.pointsomacell import PointSomaCell
from bmtk.simulator.bionet.virtualcell import VirtualCell
from bmtk.simulator.bionet.morphology import Morphology
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet import nrn
from bmtk.simulator.bionet.sonata_adaptors import BioNodeAdaptor, BioEdgeAdaptor
from .gids import GidPool

# TODO: leave this import, it will initialize some of the default functions for building neurons/synapses/weights.
import bmtk.simulator.bionet.default_setters


pc = h.ParallelContext()  # object to access MPI methods
MPI_size = int(pc.nhost())
MPI_rank = int(pc.id())


class BioNetwork(SimNetwork):
    model_type_col = 'model_type'

    def __init__(self):
        super(BioNetwork, self).__init__()
        self._io = io

        # TODO: Find a better way that will allow users to register their own class
        self._model_type_map = {
            'biophysical': BioCell,
            'point_process': PointProcessCell,
            'point_neuron': PointProcessCell,
            'point_soma': PointSomaCell,
            'virtual': VirtualCell
        }

        # self._morphologies_cache = {}
        # self._morphology_lookup = {}

        self._rank_node_gids = {}
        self._rank_node_ids = {}
        self._rank_nodes_by_model = {m_type: {} for m_type in self._model_type_map.keys()}
        self._remote_node_cache = {}
        self._virtual_nodes = {}
        self._disconnected_source_cells = {}

        self._cells_built = False
        self._connections_initialized = False

        self._gid_pool = GidPool()

        self.has_spont_syns = False
        self.spont_syns_filter_pre = None
        self.spont_syns_filter_post = None
        self.spont_syns_times = None

    @property
    def gid_pool(self):
        return self._gid_pool

    @property
    def py_function_caches(self):
        return nrn

    def set_spont_syn_activity(self, precell_filter, postcell_filter, timestamps):
        self._model_type_map = {
            'biophysical': BioCellSpontSyn,
            'point_process': PointProcessCellSpontSyns,
            'point_neuron': PointProcessCellSpontSyns,
            'point_soma': PointSomaCell,
            'virtual': VirtualCell
        }

        self.has_spont_syns = True
        self.spont_syns_filter_pre = precell_filter
        self.spont_syns_filter_post = postcell_filter
        self.spont_syns_times = timestamps

    def get_node_id(self, population, node_id):
        if node_id in self._rank_node_ids[population]:
            return self._rank_node_ids[population][node_id].node

        elif node_id in self._remote_node_cache[population]:
            return self._remote_node_cache[population][node_id]

        else:
            node_pop = self.get_node_population(population)
            node = node_pop.get_node(node_id)
            self._remote_node_cache[population][node_id] = node
            return node

    def cell_type_maps(self, model_type):
        return self._rank_nodes_by_model[model_type]

    def get_cell_node_id(self, population, node_id):
        return self._rank_node_ids[population].get(node_id, None)

    def get_cell_gid(self, gid):
        return self._rank_node_gids[gid]

    def get_local_cells(self):
        return self._rank_node_gids

    @property
    def local_gids(self):
        return list(self._rank_node_gids.keys())

    def add_nodes(self, node_population):
        self._gid_pool.add_pool(node_population.name, node_population.node_ids)
        super(BioNetwork, self).add_nodes(node_population)

    def get_virtual_cells(self, population, node_id, spike_trains, spikes_generator=None, sim=None):
        if node_id in self._virtual_nodes[population]:
            return self._virtual_nodes[population][node_id]
        else:
            node = self.get_node_id(population, node_id)
            virt_cell = VirtualCell(node, population, spike_trains, spikes_generator, sim)
            self._virtual_nodes[population][node_id] = virt_cell
            return virt_cell

    def get_disconnected_cell(self, population, node_id, spike_trains):
        if population not in self._disconnected_source_cells:
            self._disconnected_source_cells[population] = {}

        if node_id in self._disconnected_source_cells[population]:
            virt_cell = self._disconnected_source_cells[population][node_id]
        else:
            node = self.get_node_id(population, node_id)
            virt_cell = VirtualCell(node, population, spike_trains, self)
            self._disconnected_source_cells[population][node_id] = virt_cell

        return virt_cell

    def _build_cell(self, bionode, population_name):
        if bionode.model_type in self._model_type_map:
            cell = self._model_type_map[bionode.model_type](bionode, population_name=population_name, bionetwork=self)
            self._rank_nodes_by_model[bionode.model_type][cell.gid] = cell
            return cell
        else:
            self.io.log_exception('Unrecognized model_type {}.'.format(bionode.model_type))

    def _register_adaptors(self):
        super(BioNetwork, self)._register_adaptors()
        self._node_adaptors['sonata'] = BioNodeAdaptor
        self._edge_adaptors['sonata'] = BioEdgeAdaptor

    def build_nodes(self):
        for node_pop in self.node_populations:
            self._remote_node_cache[node_pop.name] = {}
            node_ids_map = {}
            if node_pop.internal_nodes_only:
                for node in node_pop[MPI_rank::MPI_size]:
                    cell = self._build_cell(bionode=node, population_name=node_pop.name)
                    node_ids_map[node.node_id] = cell
                    self._rank_node_gids[cell.gid] = cell

            elif node_pop.mixed_nodes:
                # node population contains both internal and virtual (external) nodes and the virtual nodes must be
                # filtered out
                self._virtual_nodes[node_pop.name] = {}
                for node in node_pop[MPI_rank::MPI_size]:
                    if node.model_type == 'virtual':
                        continue
                    else:
                        cell = self._build_cell(bionode=node, population_name=node_pop.name)
                        node_ids_map[node.node_id] = cell


                        self._rank_node_gids[cell.gid] = cell

            elif node_pop.virtual_nodes_only:
                self._virtual_nodes[node_pop.name] = {}

            self._rank_node_ids[node_pop.name] = node_ids_map

        # self.make_morphologies()
        # self.set_seg_props()  # set segment properties by creating Morphologies
        # self.calc_seg_coords()  # use for computing the ECP
        self._cells_built = True
        self.io.barrier()

    # def set_seg_props(self):
    #     """Set morphological properties for biophysically (morphologically) detailed cells"""
    #     for _, morphology in self._morphologies_cache.items():
    #         morphology.set_seg_props()

    # def calc_seg_coords(self):
    #     """Needed for the ECP calculations"""
    #     # TODO: Is there any reason this function can't be moved to make_morphologies()
    #     for morphology_file, morphology in self._morphologies_cache.items():
    #         morph_seg_coords = morphology.calc_seg_coords()   # needed for ECP calculations
    #
    #         for gid in self._morphology_lookup[morphology_file]:
    #             self.get_cell_gid(gid).calc_seg_coords(morph_seg_coords)

    # def make_morphologies(self):
    #     """Creating a Morphology object for each biophysical model"""
    #     # TODO: Let Morphology take care of the cache
    #     # TODO: Let other types have morphologies
    #     # TODO: Get all available morphologies from TypesTable or group
    #     for gid, cell in self._rank_node_gids.items():
    #         if not isinstance(cell, BioCell):
    #             continue
    #
    #         morphology_file = cell.morphology_file
    #         if morphology_file in self._morphologies_cache:
    #             # create a single morphology object for each model_group which share that morphology
    #             morph = self._morphologies_cache[morphology_file]
    #
    #             # associate morphology with a cell
    #             cell.set_morphology(morph)
    #             self._morphology_lookup[morphology_file].append(cell.gid)
    #
    #         else:
    #             hobj = cell.hobj  # get hoc object (hobj) from the first cell with a new morphologys
    #             morph = Morphology(hobj)
    #
    #             # associate morphology with a cell
    #             cell.set_morphology(morph)
    #
    #             # create a single morphology object for each model_group which share that morphology
    #             self._morphologies_cache[morphology_file] = morph
    #             self._morphology_lookup[morphology_file] = [cell.gid]
    #
    #     self.io.barrier()

    def _init_connections(self):
        if not self._connections_initialized:
            for gid, cell in self._rank_node_gids.items():
                cell.init_connections()
            self._connections_initialized = True

    def get_gj_id(self, network, src_nid, trg_nid, source_gap):
        '''
        Returns the gap junction id for the given nodes on a given network.

        :param source_gap: whether to return the id of the gap junction on the source node or the target node.
        '''
        if src_nid == trg_nid:
            raise Exception("Cells cannot have gap junctions with themselves.")

        gap_ids = self._gap_juncs[network]

        loc = np.where(np.logical_and(gap_ids["source_ids"] == src_nid, gap_ids["target_ids"] == trg_nid))[0]

        if len(loc) > 1:
            raise Exception("The gap junction file has more than one gap junction with the same ids.")
        elif len(loc) == 0:
            raise Exception("The gap junction file does not contain a gap junction with source id " +
             str(src_nid) + " and target id " + str(trg_nid) + " on network " + network)

        ids = [gap_ids["src_gap_ids"][loc[0]], gap_ids["trg_gap_ids"][loc[0]]]

        if not source_gap:
            ids.reverse()
        
        return ids

    def build_recurrent_edges(self):
        recurrent_edge_pops = [ep for ep in self._edge_populations if not ep.virtual_connections]
        if not recurrent_edge_pops:
            return

        self._init_connections()
        for edge_pop in recurrent_edge_pops:
            if edge_pop.recurrent_connections:
                source_population = edge_pop.source_nodes
                for trg_nid, trg_cell in self._rank_node_ids[edge_pop.target_nodes].items():
                    for edge in edge_pop.get_target(trg_nid):
                        src_node = self.get_node_id(source_population, edge.source_node_id)

                        if edge.is_gap_junction:
                            if source_population != edge_pop.target_nodes:
                                raise Exception("Gap junctions must be from the same network builder")
                            gj_ids = self.get_gj_id(source_population, edge.source_node_id, trg_nid, False)
                            trg_cell.set_syn_connection(edge, src_node, gj_ids=gj_ids)
                        else:
                            trg_cell.set_syn_connection(edge, src_node)
                for src_nid, src_cell in self._rank_node_ids[source_population].items():
                    for edge in edge_pop.get_source(src_nid):
                        if edge.is_gap_junction:
                            if source_population != edge_pop.target_nodes:
                                raise Exception("Gap junctions must be from the same network builder")

                            trg_node = self.get_node_id(edge_pop.target_nodes, edge.target_node_id)

                            gj_ids = self.get_gj_id(source_population, src_nid, edge.target_node_id, True)

                            src_cell.set_syn_connection(edge, trg_node, gj_ids=gj_ids)

            elif edge_pop.mixed_connections:
                # When dealing with edges that contain both virtual and recurrent edges we have to check every source
                # node to see if is virtual (bc virtual nodes can't be built yet). This conditional can significantly
                # slow down build time so we use a special loop that can be ignored.
                source_population = edge_pop.source_nodes
                for trg_nid, trg_cell in self._rank_node_ids[edge_pop.target_nodes].items():
                    for edge in edge_pop.get_target(trg_nid):
                        src_node = self.get_node_id(source_population, edge.source_node_id)
                        if src_node.model_type == 'virtual':
                            continue
                        trg_cell.set_syn_connection(edge, src_node)

        self.io.barrier()

    def build_replay_inputs(self, spike_trains, edges_path, edge_types_path, source_node_set, target_node_set):
        self._init_connections()

        src_cells = self.get_node_set(source_node_set)
        valid_src_ids = set(src_cells.gids())
        trg_cells = self.get_node_set(target_node_set)
        valid_trg_ids = set(trg_cells.gids())

        edges_pop_list = sonata_reader.load_edges(edges_path, edge_types_path, adaptor=self.get_edge_adaptor('sonata'))
        for edges_pop in edges_pop_list:
            edges_pop.initialize(self)

            trg_population = edges_pop.target_nodes
            src_population = edges_pop.source_nodes
            if src_population not in src_cells.population_names() or trg_population not in trg_cells.population_names():
                continue

            for trg_nid, trg_cell in self._rank_node_ids[trg_population].items():
                if trg_nid not in valid_trg_ids:
                    continue

                for edge in edges_pop.get_target(trg_nid):
                    src_nid = edge.source_node_id
                    if src_nid not in valid_src_ids:
                        continue

                    src_cell = self.get_disconnected_cell(src_population, src_nid, spike_trains)
                    trg_cell.set_syn_connection(edge, src_cell, src_cell)

        self.io.barrier()

    def find_edges(self, source_nodes=None, target_nodes=None):
        selected_edges = self._edge_populations[:]

        if source_nodes is not None:
            selected_edges = [edge_pop for edge_pop in selected_edges if edge_pop.source_nodes == source_nodes]

        if target_nodes is not None:
            selected_edges = [edge_pop for edge_pop in selected_edges if edge_pop.target_nodes == target_nodes]

        return selected_edges

    def add_spike_trains(self, spike_trains, node_set, spikes_generator=None, sim=None):
        self._init_connections()

        src_nodes = [node_pop for node_pop in self.node_populations if node_pop.name in node_set.population_names()]
        for src_node_pop in src_nodes:
            source_population = src_node_pop.name
            for edge_pop in self.find_edges(source_nodes=source_population):
                if edge_pop.virtual_connections:
                    for trg_nid, trg_cell in self._rank_node_ids[edge_pop.target_nodes].items():
                        for edge in edge_pop.get_target(trg_nid):
                            src_cell = self.get_virtual_cells(source_population, edge.source_node_id, spike_trains, spikes_generator, sim)
                            trg_cell.set_syn_connection(edge, src_cell, src_cell)

                elif edge_pop.mixed_connections:
                    raise NotImplementedError()

        self.io.barrier()

    INTFIRE_ATTRS = ['tau', 'refrac', 'm', 'taum', 'taus', 'ib', 'i', 'I', 'taue', 'taui1', 'taui2', 'i1', 'i2']

    def inspect_cells(self, format='json', output_path=None, filter=None):
        import pandas as pd
        import json
        from pathlib import Path
        from pprint import pprint

        cell_mechs = {
            'population': [],
            'node_id': [],
            'model_type': [],
            'node_type_id': [],
            'sec_name': [],
            'attr_name': [],
            'mech_name': [],
            'attr_val': [],
            'type': []
        }

        sect_counts = {
            'population': [],
            'node_id': [],
            'nsections': [],
            'nsegments': []
        }

        
        filtered_gids = set()
        filter = filter or 'all'
        for a in self.get_node_set(filter).fetch_nodes():
            filtered_gids.add((a.node_id, a.population_name))

        for _, cell in self.get_local_cells().items():
            if (cell.node_id, cell.population_name) not in filtered_gids:
                continue

            if cell['model_type'] == 'biophysical':
                sect_counts['population'].append(cell.population_name)
                sect_counts['node_id'].append(cell.node_id)
                sect_counts['nsections'].append(sum(1 for _ in cell.hobj.all))
                sect_counts['nsegments'].append(sum(sec.nseg for sec in cell.hobj.all))
                
                for sec in cell.hobj.all:
                    sec_name = sec.name().split('.')[-1].split('[')[0]
                    if hasattr(sec, 'Ra'):
                        cell_mechs['population'].append(cell.population_name)
                        cell_mechs['node_id'].append(cell.node_id)
                        cell_mechs['node_type_id'].append(cell['node_type_id'])
                        cell_mechs['model_type'].append(cell['model_type'])
                        cell_mechs['sec_name'].append(sec_name)
                        cell_mechs['mech_name'].append(None)
                        cell_mechs['attr_name'].append('Ra')
                        cell_mechs['attr_val'].append(sec.Ra)
                        cell_mechs['type'].append(None)

                    for mech_name, mech_props in sec.psection()['density_mechs'].items():
                        for attr_name, attr_val in mech_props.items():
                            cell_mechs['population'].append(cell.population_name)
                            cell_mechs['node_id'].append(cell.node_id)
                            cell_mechs['node_type_id'].append(cell['node_type_id'])
                            cell_mechs['model_type'].append(cell['model_type'])
                            cell_mechs['sec_name'].append(sec_name)
                            cell_mechs['mech_name'].append(mech_name)
                            cell_mechs['attr_name'].append(attr_name)
                            cell_mechs['attr_val'].append(np.mean(attr_val))
                            cell_mechs['type'].append('mechanism')

                    for ion_name, ion_props in sec.psection()['ions'].items():
                        for attr_name, attr_val in ion_props.items():
                            cell_mechs['population'].append(cell.population_name)
                            cell_mechs['node_id'].append(cell.node_id)
                            cell_mechs['node_type_id'].append(cell['node_type_id'])
                            cell_mechs['model_type'].append(cell['model_type'])
                            cell_mechs['sec_name'].append(sec_name)
                            cell_mechs['mech_name'].append(ion_name)
                            cell_mechs['attr_name'].append(attr_name)
                            cell_mechs['attr_val'].append(np.mean(attr_val))
                            cell_mechs['type'].append('ion')
       
            elif cell['model_type'] in ['point_neuron', 'point', 'point_process']:
                process_name = sec.name().split('.')[-1].split('[')[0]
                for attr_name in dir(cell.hobj):
                    if attr_name.startswith('__') or attr_name not in self.INTFIRE_ATTRS:
                        continue

                    cell_mechs['population'].append(cell.population_name)
                    cell_mechs['node_id'].append(cell.node_id)
                    cell_mechs['node_type_id'].append(cell['node_type_id'])
                    cell_mechs['model_type'].append(cell['model_type'])
                    cell_mechs['sec_name'].append('NA')
                    cell_mechs['mech_name'].append(process_name)
                    cell_mechs['attr_name'].append(attr_name)
                    cell_mechs['attr_val'].append(getattr(cell.hobj, attr_name))
                    cell_mechs['type'].append('ARTIFICIAL_CELL')       

        agg_mech = pd.DataFrame(cell_mechs).groupby(['population', 'node_id', 'node_type_id', 'model_type', 'sec_name', 'mech_name', 'attr_name', 'type']).agg('mean')
        mechs_df = agg_mech.reset_index()
        mechs_df['attr_name'] = mechs_df.apply(lambda r: f'{r["attr_name"]}_{r["mech_name"]}' if r['type'] == 'mechanism' else r['attr_name'], axis=1)       

        sect_lu = pd.DataFrame(sect_counts).set_index(['population', 'node_id'])

        if format == 'csv':
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                mechs_df.to_csv(output_path, index=False)
            else:
                csv_str = mechs_df.to_csv(index=False)
                print(csv_str)

        else:
            mechs_dict = {}
            for pop, pop_df in mechs_df.groupby('population'):
                mechs_dict[pop] = {}
                for node_id, node_df in pop_df.groupby('node_id'):
                    mechs_dict[pop][node_id] = {
                        'node_type_id': int(node_df['node_type_id'].iloc[0]),
                        'model_type': node_df['model_type'].iloc[0],
                        'nsections': int(sect_lu.loc[(pop, node_id)]['nsections']),
                        'nsegments': int(sect_lu.loc[(pop, node_id)]['nsegments'])
                    }
                    for sec, sec_df in node_df.groupby('sec_name'):
                        mechs_dict[pop][node_id][sec] = {}
                        for _, r in sec_df.iterrows():
                            mechs_dict[pop][node_id][sec][r['attr_name']] = {'default_value': float(r['attr_val']), 'mechanism': r['mech_name'], 'type': r['type']}

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(mechs_dict, f, indent=2)
            else:
                pprint(mechs_dict)
