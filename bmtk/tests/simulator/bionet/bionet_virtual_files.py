import numpy as np

from bmtk.utils.io import tabular_network as tn


class NodeRow(tn.NodeRow):
    @property
    def with_dynamics_params(self):
        return False


class EdgeRow(tn.EdgeRow):
    @property
    def with_dynamics_params(self):
        return False


class NodesFile(object):
    def __init__(self, N):
        self._network_name = 'test_bionet'
        self._version = None
        self._iter_index = 0
        self._nrows = 0
        self._node_types_table = None

        self._N = N
        self._rot_delta = 360.0/float(N)
        self._node_types_table = {
            101: {
                'pop_name': 'bio_exc', 'node_type_id': 101, 'model_type': 'biophysical',
                'morphology': 'be_morphology.swc',
                'dynamics_params': 'be_dynamics.json',
                'ei': 'e'
            },
            102: {
                'pop_name': 'point_exc', 'node_type_id': 102, 'model_type': 'point_IntFire1',
                'dynamics_params': 'pe_dynamics.json',
                'ei': 'e'
            },
            103: {
                'pop_name': 'bio_inh', 'node_type_id': 103, 'model_type': 'biophysical',
                'morphology': 'bi_morphology.swc',
                'dynamics_params': 'bi_dynamics.json',
                'ei': 'i'
            },
            104: {
                'pop_name': 'point_inh', 'node_type_id': 104, 'model_type': 'point_IntFire1',
                'dynamics_params': 'bi_dynamics.json',
                'ei': 'i'
            }

        }


    @property
    def name(self):
        """name of network containing these nodes"""
        return self._network_name

    @property
    def version(self):
        return self._version

    @property
    def gids(self):
        raise NotImplementedError()

    @property
    def node_types_table(self):
        return self._node_types_table

    def load(self, nodes_file, node_types_file):
        raise NotImplementedError()

    def get_node(self, gid, cache=False):
        return self[gid]

    def __len__(self):
        return self._N

    def __iter__(self):
        self._iter_index = 0
        return self

    def next(self):
        if self._iter_index >= len(self):
            raise StopIteration

        node_row = self[self._iter_index]
        self._iter_index += 1
        return node_row

    def __getitem__(self, gid):
        node_props = {'positions': np.random.rand(3), 'rotation': self._rot_delta*gid, 'weight': 0.0001*gid}
        return NodeRow(gid, node_props, self.__get_node_type_props(gid))


    def __get_node_type_props(self, gid):
        if gid <= self._N/4:
            return self._node_types_table[101]
        elif gid <= self._N/2:
            return self._node_types_table[102]
        elif gid <= self._N*3/4:
            return self._node_types_table[103]
        else:
            return self._node_types_table[104]


class EdgesFile(tn.EdgesFile):
    def __init__(self, target_nodes, source_nodes):
        self._target_nodes = target_nodes
        self._source_nodes = source_nodes
        self._edge_type_props = [
            {
                'node_type_id': 1,
                'target_query': 'model_type="biophysical"', 'source_query': 'ei="e"',
                'syn_weight': .10,
                'syn_targets': ['dend', 'apical'],
                'dynamics_params': 'biophys_exc.json'
            },
            {
                'node_type_id': 2,
                'target_query': 'model_type="point_IntFire1"', 'source_query': 'ei="e"',
                'syn_weight': .20,
                'dynamics_params': 'point_exc.json'
            },
            {
                'node_type_id': 3,
                'target_query': 'model_type="biophysical"', 'source_query': 'ei="i"',
                'syn_weight': -.10,
                'syn_targets': ['soma', 'dend'],
                'dynamics_params': 'biophys_inh.json'
            },
            {
                'node_type_id': 4,
                'target_query': 'model_type="point_IntFire1"', 'source_query': 'ei="i"',
                'syn_weight': -.20,
                'dynamics_params': 'point_inh.json'
            }
        ]


    @property
    def source_network(self):
        """Name of network containing the source gids"""
        return self._source_nodes.name

    @property
    def target_network(self):
        """Name of network containing the target gids"""
        return self._target_nodes.name

    def load(self, edges_file, edge_types_file):
        raise NotImplementedError()

    def edges_itr(self, target_gid):
        trg_node = self._target_nodes[target_gid]
        for src_node in self._source_nodes:
            edge_props = {'syn_weight': trg_node['weight']}
            #edge_type_props = {'edge_type_id': 1}
            yield EdgeRow(trg_node.gid, src_node.gid, edge_props, self.__get_edge_type_prop(src_node, trg_node))

        #def __init__(self, trg_gid, src_gid, edge_props={}, edge_type_props={}):
        #raise NotImplementedError()

    def __len__(self):
        return len(self._source_nodes)*len(self._target_nodes)

    def __get_edge_type_prop(self, source_node, target_node):
        indx = 0 if source_node['ei'] == 'e' else 2
        indx += 0 if target_node['model_type'] == 'biophysical' else 1
        return self._edge_type_props[indx]
