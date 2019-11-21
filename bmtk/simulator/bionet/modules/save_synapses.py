import os
import csv
import h5py
import numpy as np
from neuron import h
from glob import glob
from itertools import product

from .sim_module import SimulatorMod
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.io_tools import io
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version
from bmtk.simulator.bionet.pointprocesscell import PointProcessCell


pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class SaveSynapses(SimulatorMod):
    def __init__(self, network_dir, single_file=False, **params):
        self._network_dir = network_dir
        self._virt_lookup = {}
        self._gid_lookup = {}
        self._sec_lookup = {}
        if MPI_RANK == 0:
            if not os.path.exists(network_dir):
                os.makedirs(network_dir)
        pc.barrier()

        #if N_HOSTS > 1:
        #    io.log_exception('save_synapses module is not current supported with mpi')

        self._syn_writer = ConnectionWriter(network_dir)

    def _print_nc(self, nc, src_nid, trg_nid, cell, src_pop, trg_pop, edge_type_id):
        if isinstance(cell, BioCell):
            sec_x = nc.postloc()
            sec = h.cas()
            sec_id = self._sec_lookup[cell.gid][sec]  # cell.get_section_id(sec)
            h.pop_section()
            self._syn_writer.add_bio_conn(edge_type_id, src_nid, src_pop, trg_nid, trg_pop, nc.weight[0], sec_id, sec_x)

        else:
            self._syn_writer.add_point_conn(edge_type_id, src_nid, src_pop, trg_nid, trg_pop, nc.weight[0])

    def initialize(self, sim):
        io.log_info('Saving network connections. This may take a while.')

        # Need a way to look up virtual nodes from nc.pre()
        for pop_name, nodes_table in sim.net._virtual_nodes.items():
            for node_id, virt_node in nodes_table.items():
                self._virt_lookup[virt_node.hobj] = (pop_name, node_id)

        # Need to figure out node_id and pop_name from nc.srcgid()
        for node_pop in sim.net.node_populations:
            pop_name = node_pop.name
            for node in node_pop[0::1]:
                if node.model_type != 'virtual':
                    self._gid_lookup[node.gid] = (pop_name, node.node_id)

        for gid, cell in sim.net.get_local_cells().items():
            trg_pop, trg_id = self._gid_lookup[gid]
            if isinstance(cell, BioCell):
                # sections = cell._syn_seg_ix
                self._sec_lookup[gid] = {sec_name: sec_id for sec_id, sec_name in enumerate(cell.get_sections_id())}

            else:
                sections = [-1]*len(cell.netcons)

            for nc, edge_type_id in zip(cell.netcons, cell._edge_type_ids):
                src_gid = int(nc.srcgid())
                if src_gid == -1:
                    # source is a virtual node
                    src_pop, src_id = self._virt_lookup[nc.pre()]
                else:
                    src_pop, src_id = self._gid_lookup[src_gid]

                self._print_nc(nc, src_id, trg_id, cell, src_pop, trg_pop, edge_type_id)

        self._syn_writer.close()
        pc.barrier()

        if N_HOSTS > 1 and MPI_RANK == 0:
            merger = H5Merger(self._network_dir, self._syn_writer._pop_groups.keys())

        pc.barrier()

        io.log_info('    Done saving network connections.')


class H5Merger(object):
    def __init__(self, network_dir, grp_keys):
        self._network_dir = network_dir
        self._grp_keys = list(grp_keys)

        self._edge_counts = {(s, t): 0 for s, t in self._grp_keys}
        self._biophys_edge_count = {(s, t): 0 for s, t in self._grp_keys}
        self._point_edge_count = {(s, t): 0 for s, t in self._grp_keys}

        self._tmp_files = {(s, t): [] for s, t in self._grp_keys}
        for (src_pop, trg_pop), r in product(self._grp_keys, range(N_HOSTS)):
            fname = '.core{}.{}_{}_edges.h5'.format(r, src_pop, trg_pop)
            fpath = os.path.join(self._network_dir, fname)
            if not os.path.exists(fpath):
                io.log_warning('Expected file {} is missing'.format(fpath))

            h5file = h5py.File(fpath, 'r')
            edges_grp = h5file['/edges/{}_{}'.format(src_pop, trg_pop)]
            self._tmp_files[(src_pop, trg_pop)].append(edges_grp)

            self._edge_counts[(src_pop, trg_pop)] += len(edges_grp['source_node_id'])
            self._biophys_edge_count[(src_pop, trg_pop)] += len(edges_grp['0/syn_weight'])
            self._point_edge_count[(src_pop, trg_pop)] += len(edges_grp['1/syn_weight'])

        for (src_pop, trg_pop), in_grps in self._tmp_files.items():
            out_h5 = h5py.File(os.path.join(self._network_dir, '{}_{}_edges.h5'.format(src_pop, trg_pop)), 'w')
            add_hdf5_magic(out_h5)
            add_hdf5_version(out_h5)
            pop_root = out_h5.create_group('/edges/{}_{}'.format(src_pop, trg_pop))
            n_edges_total = self._edge_counts[(src_pop, trg_pop)]
            n_edges_bio = self._biophys_edge_count[(src_pop, trg_pop)]
            n_edges_point = self._point_edge_count[(src_pop, trg_pop)]

            pop_root.create_dataset('source_node_id', (n_edges_total, ), dtype=np.uint64)
            pop_root['source_node_id'].attrs['node_population'] = src_pop

            pop_root.create_dataset('target_node_id', (n_edges_total, ), dtype=np.uint64)
            pop_root['target_node_id'].attrs['node_population'] = trg_pop

            pop_root.create_dataset('edge_group_id', (n_edges_total, ), dtype=np.uint16)
            pop_root.create_dataset('edge_group_index', (n_edges_total,), dtype=np.uint16)
            pop_root.create_dataset('edge_type_id', (n_edges_total, ), dtype=np.uint32)

            pop_root.create_dataset('0/syn_weight', (n_edges_bio, ), dtype=np.float)
            pop_root.create_dataset('0/sec_id', (n_edges_bio, ), dtype=np.uint64)
            pop_root.create_dataset('0/sec_x', (n_edges_bio, ), dtype=np.float)
            pop_root.create_dataset('1/syn_weight', (n_edges_point, ), dtype=np.float)

            total_offset = 0
            bio_offset = 0
            point_offset = 0
            for grp in in_grps:
                n_ds = len(grp['source_node_id'])
                pop_root['source_node_id'][total_offset:(total_offset + n_ds)] = grp['source_node_id'][()]
                pop_root['target_node_id'][total_offset:(total_offset + n_ds)] = grp['target_node_id'][()]
                pop_root['edge_group_id'][total_offset:(total_offset + n_ds)] = grp['edge_group_id'][()]
                pop_root['edge_group_index'][total_offset:(total_offset + n_ds)] = grp['edge_group_index'][()]
                pop_root['edge_type_id'][total_offset:(total_offset + n_ds)] = grp['edge_type_id'][()]
                total_offset += n_ds

                n_ds = len(grp['0/syn_weight'])
                # print(grp['0/syn_weight'][()])
                pop_root['0/syn_weight'][bio_offset:(bio_offset + n_ds)] = grp['0/syn_weight'][()]
                pop_root['0/sec_id'][bio_offset:(bio_offset + n_ds)] = grp['0/sec_id'][()]
                pop_root['0/sec_x'][bio_offset:(bio_offset + n_ds)] = grp['0/sec_x'][()]
                bio_offset += n_ds

                n_ds = len(grp['1/syn_weight'])
                pop_root['1/syn_weight'][point_offset:(point_offset + n_ds)] = grp['1/syn_weight'][()]
                point_offset += n_ds

                fname = grp.file.filename
                grp.file.close()
                if os.path.exists(fname):
                    os.remove(fname)

            self._create_index(pop_root, index_type='target')
            self._create_index(pop_root, index_type='source')
            out_h5.close()

    def _create_index(self, pop_root, index_type='target'):
        if index_type == 'target':
            edge_nodes = np.array(pop_root['target_node_id'], dtype=np.int64)
            output_grp = pop_root.create_group('indices/target_to_source')
        elif index_type == 'source':
            edge_nodes = np.array(pop_root['source_node_id'], dtype=np.int64)
            output_grp = pop_root.create_group('indices/source_to_target')

        edge_nodes = np.append(edge_nodes, [-1])
        n_targets = np.max(edge_nodes)
        ranges_list = [[] for _ in range(n_targets + 1)]

        n_ranges = 0
        begin_index = 0
        cur_trg = edge_nodes[begin_index]
        for end_index, trg_gid in enumerate(edge_nodes):
            if cur_trg != trg_gid:
                ranges_list[cur_trg].append((begin_index, end_index))
                cur_trg = int(trg_gid)
                begin_index = end_index
                n_ranges += 1

        node_id_to_range = np.zeros((n_targets + 1, 2))
        range_to_edge_id = np.zeros((n_ranges, 2))
        range_index = 0
        for node_index, trg_ranges in enumerate(ranges_list):
            if len(trg_ranges) > 0:
                node_id_to_range[node_index, 0] = range_index
                for r in trg_ranges:
                    range_to_edge_id[range_index, :] = r
                    range_index += 1
                node_id_to_range[node_index, 1] = range_index

        output_grp.create_dataset('range_to_edge_id', data=range_to_edge_id, dtype='uint64')
        output_grp.create_dataset('node_id_to_range', data=node_id_to_range, dtype='uint64')




class ConnectionWriter(object):
    class H5Index(object):
        def __init__(self, file_path, src_pop, trg_pop):
            # TODO: Merge with NetworkBuilder code for building SONATA files
            self._nsyns = 0
            self._n_biosyns = 0
            self._n_pointsyns = 0
            self._block_size = 5

            self._pop_name = '{}_{}'.format(src_pop, trg_pop)
            # self._h5_file = h5py.File(os.path.join(network_dir, '{}_edges.h5'.format(self._pop_name)), 'w')
            self._h5_file = h5py.File(file_path, 'w')
            add_hdf5_magic(self._h5_file)
            add_hdf5_version(self._h5_file)
            self._pop_root = self._h5_file.create_group('/edges/{}'.format(self._pop_name))
            self._pop_root.create_dataset('edge_group_id', (self._block_size, ), dtype=np.uint16,
                                          chunks=(self._block_size, ), maxshape=(None, ))
            self._pop_root.create_dataset('source_node_id', (self._block_size, ), dtype=np.uint64,
                                          chunks=(self._block_size, ), maxshape=(None, ))
            self._pop_root['source_node_id'].attrs['node_population'] = src_pop
            self._pop_root.create_dataset('target_node_id', (self._block_size, ), dtype=np.uint64,
                                          chunks=(self._block_size, ), maxshape=(None, ))
            self._pop_root['target_node_id'].attrs['node_population'] = trg_pop
            self._pop_root.create_dataset('edge_type_id', (self._block_size, ), dtype=np.uint32,
                                          chunks=(self._block_size, ), maxshape=(None, ))
            self._pop_root.create_dataset('0/syn_weight', (self._block_size, ), dtype=np.float,
                                          chunks=(self._block_size, ), maxshape=(None, ))
            self._pop_root.create_dataset('0/sec_id', (self._block_size, ), dtype=np.uint64,
                                          chunks=(self._block_size, ), maxshape=(None, ))
            self._pop_root.create_dataset('0/sec_x', (self._block_size, ), chunks=(self._block_size, ),
                                          maxshape=(None, ), dtype=np.float)
            self._pop_root.create_dataset('1/syn_weight', (self._block_size, ), dtype=np.float,
                                          chunks=(self._block_size, ), maxshape=(None, ))

        def _add_conn(self, edge_type_id, src_id, trg_id, grp_id):
            self._pop_root['edge_type_id'][self._nsyns] = edge_type_id
            self._pop_root['source_node_id'][self._nsyns] = src_id
            self._pop_root['target_node_id'][self._nsyns] = trg_id
            self._pop_root['edge_group_id'][self._nsyns] = grp_id

            self._nsyns += 1
            if self._nsyns % self._block_size == 0:
                self._pop_root['edge_type_id'].resize((self._nsyns + self._block_size,))
                self._pop_root['source_node_id'].resize((self._nsyns + self._block_size, ))
                self._pop_root['target_node_id'].resize((self._nsyns + self._block_size, ))
                self._pop_root['edge_group_id'].resize((self._nsyns + self._block_size, ))

        def add_bio_conn(self, edge_type_id, src_id, trg_id, syn_weight, sec_id, sec_x):
            self._add_conn(edge_type_id, src_id, trg_id, 0)
            self._pop_root['0/syn_weight'][self._n_biosyns] = syn_weight
            self._pop_root['0/sec_id'][self._n_biosyns] = sec_id
            self._pop_root['0/sec_x'][self._n_biosyns] = sec_x

            self._n_biosyns += 1
            if self._n_biosyns % self._block_size == 0:
                self._pop_root['0/syn_weight'].resize((self._n_biosyns + self._block_size, ))
                self._pop_root['0/sec_id'].resize((self._n_biosyns + self._block_size, ))
                self._pop_root['0/sec_x'].resize((self._n_biosyns + self._block_size, ))

        def add_point_conn(self, edge_type_id, src_id, trg_id, syn_weight):
            self._add_conn(edge_type_id, src_id, trg_id, 1)
            self._pop_root['1/syn_weight'][self._n_pointsyns] = syn_weight

            self._n_pointsyns += 1
            if self._n_pointsyns % self._block_size == 0:
                self._pop_root['1/syn_weight'].resize((self._n_pointsyns + self._block_size, ))

        def clean_ends(self):
            self._pop_root['source_node_id'].resize((self._nsyns,))
            self._pop_root['target_node_id'].resize((self._nsyns,))
            self._pop_root['edge_group_id'].resize((self._nsyns,))
            self._pop_root['edge_type_id'].resize((self._nsyns,))

            self._pop_root['0/syn_weight'].resize((self._n_biosyns,))
            self._pop_root['0/sec_id'].resize((self._n_biosyns,))
            self._pop_root['0/sec_x'].resize((self._n_biosyns,))

            self._pop_root['1/syn_weight'].resize((self._n_pointsyns,))

            eg_ds = self._pop_root.create_dataset('edge_group_index', (self._nsyns, ), dtype=np.uint64)
            bio_count, point_count = 0, 0
            for idx, grp_id in enumerate(self._pop_root['edge_group_id']):
                if grp_id == 0:
                    eg_ds[idx] = bio_count
                    bio_count += 1
                elif grp_id == 1:
                    eg_ds[idx] = point_count
                    point_count += 1

            self._create_index('target')
            self._create_index('source')

        def _create_index(self, index_type='target'):
            if index_type == 'target':
                edge_nodes = np.array(self._pop_root['target_node_id'], dtype=np.int64)
                output_grp = self._pop_root.create_group('indices/target_to_source')
            elif index_type == 'source':
                edge_nodes = np.array(self._pop_root['source_node_id'], dtype=np.int64)
                output_grp = self._pop_root.create_group('indices/source_to_target')

            edge_nodes = np.append(edge_nodes, [-1])
            n_targets = np.max(edge_nodes)
            ranges_list = [[] for _ in range(n_targets + 1)]

            n_ranges = 0
            begin_index = 0
            cur_trg = edge_nodes[begin_index]
            for end_index, trg_gid in enumerate(edge_nodes):
                if cur_trg != trg_gid:
                    ranges_list[cur_trg].append((begin_index, end_index))
                    cur_trg = int(trg_gid)
                    begin_index = end_index
                    n_ranges += 1

            node_id_to_range = np.zeros((n_targets + 1, 2))
            range_to_edge_id = np.zeros((n_ranges, 2))
            range_index = 0
            for node_index, trg_ranges in enumerate(ranges_list):
                if len(trg_ranges) > 0:
                    node_id_to_range[node_index, 0] = range_index
                    for r in trg_ranges:
                        range_to_edge_id[range_index, :] = r
                        range_index += 1
                    node_id_to_range[node_index, 1] = range_index

            output_grp.create_dataset('range_to_edge_id', data=range_to_edge_id, dtype='uint64')
            output_grp.create_dataset('node_id_to_range', data=node_id_to_range, dtype='uint64')

        def close_h5(self):
            self._h5_file.close()

    def __init__(self, network_dir):
        self._network_dir = network_dir
        self._pop_groups = {}

    def _group_key(self, src_pop, trg_pop):
        return (src_pop, trg_pop)

    def _get_edge_group(self, src_pop, trg_pop):
        grp_key = self._group_key(src_pop, trg_pop)
        if grp_key not in self._pop_groups:
            pop_name = '{}_{}'.format(src_pop, trg_pop)
            if N_HOSTS > 1:
                pop_name = '.core{}.{}'.format(MPI_RANK, pop_name)

            file_path = os.path.join(self._network_dir, '{}_edges.h5'.format(pop_name))
            self._pop_groups[grp_key] = self.H5Index(file_path, src_pop, trg_pop)

        return self._pop_groups[grp_key]

    def add_bio_conn(self, edge_type_id, src_id, src_pop, trg_id, trg_pop, syn_weight, sec_id, sec_x):
        h5_grp = self._get_edge_group(src_pop, trg_pop)
        h5_grp.add_bio_conn(edge_type_id, src_id, trg_id, syn_weight, sec_id, sec_x)

    def add_point_conn(self, edge_type_id, src_id, src_pop, trg_id, trg_pop, syn_weight):
        h5_grp = self._get_edge_group(src_pop, trg_pop)
        h5_grp.add_point_conn(edge_type_id, src_id, trg_id, syn_weight)

    def close(self):
        for _, h5index in self._pop_groups.items():
            h5index.clean_ends()
            h5index.close_h5()
