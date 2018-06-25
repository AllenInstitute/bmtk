import os
import csv
import h5py
import numpy as np
from neuron import h

from .sim_module import SimulatorMod
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.io_tools import io
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
        if not os.path.exists(network_dir):
            os.makedirs(network_dir)

        if N_HOSTS > 1:
            io.log_exception('save_synapses module is not current supported with mpi')

        self._syn_writer = ConnectionWriter(network_dir)

    def _print_nc(self, nc, src_nid, trg_nid, cell, src_pop, trg_pop):
        if isinstance(cell, BioCell):
            sec_x = nc.postloc()
            sec = h.cas()
            sec_id = self._sec_lookup[cell.gid][sec] #cell.get_section_id(sec)
            h.pop_section()
            self._syn_writer.add_bio_conn(trg_nid, trg_pop, src_nid, src_pop, nc.weight[0], sec_id, sec_x)
            #print '{} ({}) <-- {} ({}), {}, {}, {}, {}'.format(trg_nid, trg_pop, src_nid, src_pop, nc.weight[0], nc.delay, sec_id, sec_x)

        else:
            self._syn_writer.add_point_conn(trg_nid, trg_pop, src_nid, src_pop, nc.weight[0])
            # print '{} ({}) <-- {} ({}), {}, {}'.format(trg_nid, trg_pop, src_nid, src_pop, nc.weight[0], nc.delay)


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
                self._sec_lookup[gid] = {sec_name: sec_id for sec_id, sec_name in enumerate(cell.get_sections())}

            for nc in cell.netcons:
                src_gid = int(nc.srcgid())
                if src_gid == -1:
                    # source is a virtual node
                    src_pop, src_id = self._virt_lookup[nc.pre()]
                else:
                    src_pop, src_id = self._gid_lookup[src_gid]

                self._print_nc(nc, src_id, trg_id, cell, src_pop, trg_pop)

        self._syn_writer.close()
        io.log_info('Done saving network connections.')


class ConnectionWriter(object):
    class H5Index(object):
        def __init__(self, pop_root, src_pop, trg_pop):
            self._nsyns = 0
            self._n_biosyns = 0
            self._n_pointsyns = 0

            self._block_size = 5
            self._pop_root = pop_root
            self._pop_root.create_dataset('edge_group_id', (self._block_size, ), dtype=np.uint16, chunks=(self._block_size, ), maxshape=(None,))
            self._pop_root.create_dataset('source_node_id', (self._block_size, ), dtype=np.uint64, chunks=(self._block_size, ), maxshape=(None,))
            self._pop_root['source_node_id'].attrs['node_population'] = src_pop
            self._pop_root.create_dataset('target_node_id', (self._block_size, ), dtype=np.uint64, chunks=(self._block_size, ), maxshape=(None,))
            self._pop_root['target_node_id'].attrs['node_population'] = trg_pop
            self._pop_root.create_dataset('0/syn_weight', (self._block_size, ), dtype=np.float, chunks=(self._block_size, ), maxshape=(None,))
            self._pop_root.create_dataset('0/sec_id', (self._block_size, ), dtype=np.float, chunks=(self._block_size, ), maxshape=(None,))
            self._pop_root.create_dataset('0/sec_x', (self._block_size, ), dtype=np.float, chunks=(self._block_size, ), maxshape=(None,))
            self._pop_root.create_dataset('1/syn_weight', (self._block_size, ), dtype=np.float, chunks=(self._block_size, ), maxshape=(None,))

        def _add_conn(self, src_id, trg_id, grp_id):
            self._pop_root['source_node_id'][self._nsyns] = src_id
            self._pop_root['target_node_id'][self._nsyns] = trg_id
            self._pop_root['edge_group_id'][self._nsyns] = grp_id

            self._nsyns += 1
            if self._nsyns % self._block_size == 0:
                self._pop_root['source_node_id'].resize((self._nsyns + self._block_size, ))
                self._pop_root['target_node_id'].resize((self._nsyns + self._block_size, ))
                self._pop_root['edge_group_id'].resize((self._nsyns + self._block_size, ))


        def add_bio_conn(self, src_id, trg_id, syn_weight, sec_id, sec_x):
            self._add_conn(src_id, trg_id, 0)
            self._pop_root['0/syn_weight'][self._n_biosyns] = syn_weight
            self._pop_root['0/sec_id'][self._n_biosyns] = sec_id
            self._pop_root['0/sec_x'][self._n_biosyns] = sec_x

            self._n_biosyns += 1
            if self._n_biosyns % self._block_size == 0:
                self._pop_root['0/syn_weight'].resize((self._n_biosyns + self._block_size, ))
                self._pop_root['0/sec_id'].resize((self._n_biosyns + self._block_size, ))
                self._pop_root['0/sec_x'].resize((self._n_biosyns + self._block_size, ))


        def add_point_conn(self, src_id, trg_id, syn_weight):
            self._add_conn(src_id, trg_id, 1)
            self._pop_root['1/syn_weight'][self._n_pointsyns] = syn_weight

            self._n_pointsyns += 1
            if self._n_pointsyns % self._block_size == 0:
                self._pop_root['1/syn_weight'].resize((self._n_pointsyns + self._block_size, ))

        def clean_ends(self):
            self._pop_root['source_node_id'].resize((self._nsyns,))
            self._pop_root['target_node_id'].resize((self._nsyns,))
            self._pop_root['edge_group_id'].resize((self._nsyns,))

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

    def __init__(self, network_dir):
        self._network_dir = network_dir
        self._h5_file = h5py.File(os.path.join(network_dir, 'edges.h5'), 'w')
        self._edges_grp = self._h5_file.create_group('/edges')
        self._pop_groups = {}

    def _group_key(self, src_pop, trg_pop):
        return (src_pop, trg_pop)

    def _get_edge_group(self, src_pop, trg_pop):
        grp_key = self._group_key(src_pop, trg_pop)
        if grp_key not in self._pop_groups:
            grp_root = self._edges_grp.create_group('{}_to_{}'.format(src_pop, trg_pop))
            self._pop_groups[grp_key] = self.H5Index(grp_root, src_pop, trg_pop)

        return self._pop_groups[grp_key]

    def add_bio_conn(self, src_id, src_pop, trg_id, trg_pop, syn_weight, sec_id, sec_x):
        h5_grp = self._get_edge_group(src_pop, trg_pop)
        h5_grp.add_bio_conn(src_id, trg_id, syn_weight, sec_id, sec_x)

    def add_point_conn(self, src_id, src_pop, trg_id, trg_pop, syn_weight):
        h5_grp = self._get_edge_group(src_pop, trg_pop)
        h5_grp.add_point_conn(src_id, trg_id, syn_weight)

    def close(self):
        for _, h5index in self._pop_groups.items():
            h5index.clean_ends()
