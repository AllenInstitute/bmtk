import numpy as np


class NodeRow(object):
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
                'pop_name': 'internal_exc', 'node_type_id': 101, 'model_type': 'internal',
                'dynamics_params': 'exc_dynamics.json',
                'ei': 'e'
            },

            102: {
                'pop_name': 'internal_inh', 'node_type_id': 102, 'model_type': 'internal',
                'dynamics_params': 'inh_dynamics.json',
                'ei': 'i'
            },
            103: {
                'pop_name': 'external_exc', 'node_type_id': 103, 'model_type': 'external',
                'dynamics_params': 'external_dynamics.json',
                'ei': 'external_exc.json'
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
        if gid <= self._N/3:
            return self._node_types_table[101]
        elif gid <= self._N*2/3:
            return self._node_types_table[102]
        else:
            return self._node_types_table[103]


class EdgeRow(object):
    @property
    def with_dynamics_params(self):
        return False


class EdgesFile(object):
    def __init__(self, target_nodes, source_nodes):
        self._target_nodes = target_nodes
        self._source_nodes = source_nodes
        self._edge_type_props = [
            {
                'node_type_id': 1,
                'target_query': 'model_type="iaf_psc_alpha"', 'source_query': 'ei="e"',
                'syn_weight': .10,
                'delay': 2.0,
                'dynamics_params': 'iaf_exc.json'
            },
            {
                'node_type_id': 2,
                'target_query': 'model_type="iaf_psc_alpha"', 'source_query': 'ei="i"',
                'syn_weight': -.10,
                'delay': 2.0,
                'dynamics_params': 'iaf_inh.json'
            },
            {
                'node_type_id': 3,
                'target_query': 'model_type="izhikevich"', 'source_query': 'ei="e"',
                'syn_weight': .20,
                'delay': 2.0,
                'dynamics_params': 'izh_exc.json'
            },
            {
                'node_type_id': 4,
                'target_query': 'model_type="izhikevich"', 'source_query': 'ei="i"',
                'syn_weight': -.20,
                'delay': 2.0,
                'dynamics_params': 'izh_inh.json'
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
            yield EdgeRow(trg_node.gid, src_node.gid, edge_props, self.__get_edge_type_prop(src_node, trg_node))


    def __len__(self):
        return len(self._source_nodes)*len(self._target_nodes)

    def __get_edge_type_prop(self, source_node, target_node):
        indx = 0 if source_node['model_type'] == 'iaf_psc_alpha' else 2
        indx += 0 if target_node['ei'] == 'e' else 1
        return self._edge_type_props[indx]
