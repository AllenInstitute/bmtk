import numpy as np
from collections import Counter
import numbers
import nest
import types

from bmtk.simulator.core.sonata_reader import NodeAdaptor, SonataBaseNode, EdgeAdaptor, SonataBaseEdge
from bmtk.simulator.pointnet.io_tools import io
from bmtk.simulator.pointnet.pyfunction_cache import py_modules


def all_null(node_group, column_name):
    """Helper function to determine if a column has any non-NULL values"""
    types_table = node_group.parent.types_table
    non_null_vals = [types_table[ntid][column_name] for ntid in np.unique(node_group.node_type_ids)
                     if types_table[ntid][column_name] is not None]
    return len(non_null_vals) == 0


class PointNodeBatched(object):
    def __init__(self, node_ids, gids, node_types_table, node_type_id):
        self._n_nodes = len(node_ids)
        self._node_ids = node_ids
        self._gids = gids
        self._nt_table = node_types_table
        self._nt_id = node_type_id
        self._nest_ids = []

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def node_ids(self):
        return self._node_ids

    @property
    def gids(self):
        return self._gids

    @property
    def nest_ids(self):
        return self._nest_ids

    @property
    def nest_model(self):
        return self._nt_table[self._nt_id]['model_template'].split(':')[1]

    @property
    def nest_params(self):
        return self._nt_table[self._nt_id]['dynamics_params']

    @property
    def model_type(self):
        return self._nt_table[self._nt_id]['model_type']

    def build(self):
        self._nest_ids = nest.Create(self.nest_model, self.n_nodes, self.nest_params)


class PointNode(SonataBaseNode):
    def __init__(self, node, prop_adaptor):
        super(PointNode, self).__init__(node, prop_adaptor)
        self._nest_ids = []

    @property
    def n_nodes(self):
        return 1

    @property
    def node_ids(self):
        return [self._prop_adaptor.node_id(self._node)]

    @property
    def gids(self):
        return [self._prop_adaptor.gid(self._node)]

    @property
    def nest_ids(self):
        return self._nest_ids

    @property
    def nest_model(self):
        return self._prop_adaptor.model_template(self._node)[1]

    @property
    def nest_params(self):
        return self.dynamics_params

    def build(self):
        nest_model = self.nest_model
        dynamics_params = self.dynamics_params
        fnc_name = self._node['model_processing']
        if fnc_name is None:
            self._nest_ids = nest.Create(nest_model, 1, dynamics_params)
        else:
            cell_fnc = py_modules.cell_processor(fnc_name)
            self._nest_ids = cell_fnc(nest_model, self._node, dynamics_params)


class PointNodeAdaptor(NodeAdaptor):
    def __init__(self, network):
        super(PointNodeAdaptor, self).__init__(network)

        # Flag for determining if we can build multiple NEST nodes at once. If each individual node has unique
        # NEST params or a model_processing function is being called then we must nest.Create for each individual cell.
        # Otherwise we can try to call nest.Create for a batch of nodes that share the same properties
        self._can_batch = True

    @property
    def batch_process(self):
        return self._can_batch

    @batch_process.setter
    def batch_process(self, flag):
        self._can_batch = flag

    def get_node(self, sonata_node):
        return PointNode(sonata_node, self)

    def get_batches(self, node_group):
        node_ids = node_group.node_ids
        node_type_ids = node_group.node_type_ids
        node_gids = node_group.gids
        if node_gids is None:
            node_gids = node_ids

        ntids_counter = Counter(node_type_ids)

        nid_groups = {nt_id: np.zeros(ntids_counter[nt_id], dtype=np.uint32) for nt_id in ntids_counter}
        gid_groups = {nt_id: np.zeros(ntids_counter[nt_id], dtype=np.uint32) for nt_id in ntids_counter}
        node_groups_counter = {nt_id: 0 for nt_id in ntids_counter}

        for node_id, gid, node_type_id in zip(node_ids, node_gids, node_type_ids):
            grp_indx = node_groups_counter[node_type_id]
            nid_groups[node_type_id][grp_indx] = node_id
            gid_groups[node_type_id][grp_indx] = gid
            node_groups_counter[node_type_id] += 1

        return [PointNodeBatched(nid_groups[nt_id], gid_groups[nt_id], node_group.parent.node_types_table, nt_id)
                for nt_id in ntids_counter]

    @staticmethod
    def patch_adaptor(adaptor, node_group, network):
        node_adaptor = NodeAdaptor.patch_adaptor(adaptor, node_group, network)

        # If dynamics params is stored in the nodes.h5 then we have to build each node separate
        if node_group.has_dynamics_params:
            node_adaptor.batch_process = False

        # If there is a non-null value in the model_processing column then it potentially means that every cell is
        # uniquly built (currently model_processing is applied to each individ. cell) and nodes can't be batched
        if 'model_processing' in node_group.columns:
            node_adaptor.batch_process = False
        elif 'model_processing' in node_group.all_columns and not all_null(node_group, 'model_processing'):
            node_adaptor.batch_process = False

        if node_adaptor.batch_process:
            io.log_info('Batch processing nodes for {}.'.format(node_group.parent.name))

        return node_adaptor


class PointEdge(SonataBaseEdge):
    pass


class PointEdgeBatched(object):
    def __init__(self, source_nids, target_nids, nest_params):
        self._src_nids = source_nids
        self._trg_nids = target_nids
        self._nest_params = nest_params

    @property
    def source_node_ids(self):
        return self._src_nids

    @property
    def target_node_ids(self):
        return self._trg_nids

    @property
    def nest_params(self):
        return self._nest_params


class PointEdgeAdaptor(EdgeAdaptor):
    def __init__(self, network):
        super(PointEdgeAdaptor, self).__init__(network)
        self._can_batch = True

    @property
    def batch_process(self):
        return self._can_batch

    @batch_process.setter
    def batch_process(self, flag):
        self._can_batch = flag

    def synaptic_params(self, edge):
        params_dict = {'weight': self.syn_weight(edge, None, None), 'delay': edge.delay}
        params_dict.update(edge.dynamics_params)
        return params_dict

    def get_batches(self, edge_group):
        shared_properties = {}
        core_properties = {}
        src_ids = {}
        trg_ids = {}
        edge_types_table = edge_group.parent.edge_types_table

        for edge in edge_group:
            pointnet_edge = self.get_edge(edge)
            trg_nid = edge.target_node_id
            src_nid = edge.source_node_id
            et_id = edge.edge_type_id

            if et_id not in shared_properties:
                c_props = {}
                s_props = {}
                if 'model_template' in edge_types_table[et_id]:
                    c_props['model'] = edge_types_table[et_id]['model_template']
                for key, val in self.synaptic_params(pointnet_edge).items():
                    if isinstance(val, numbers.Number):
                        # TODO: Preallocate lists
                        s_props[key] = [val]
                    else:
                        c_props[key] = val

                shared_properties[et_id] = s_props
                core_properties[et_id] = c_props
                src_ids[et_id] = [src_nid]
                trg_ids[et_id] = [trg_nid]

            else:
                src_ids[et_id].append(src_nid)
                trg_ids[et_id].append(trg_nid)
                grp_dict = shared_properties[et_id]
                syn_dict = self.synaptic_params(pointnet_edge)
                for key in grp_dict.keys():
                    grp_dict[key].append(syn_dict[key])

        batched_edges = []
        for et_id in shared_properties.keys():
            shared_properties[et_id].update(core_properties[et_id])
            batched_edges.append(PointEdgeBatched(src_ids[et_id], trg_ids[et_id], shared_properties[et_id]))
        return batched_edges

    @staticmethod
    def patch_adaptor(adaptor, edge_group):
        edge_adaptor = EdgeAdaptor.patch_adaptor(adaptor, edge_group)

        if 'weight_function' not in edge_group.all_columns and 'syn_weight' in edge_group.all_columns:
            adaptor.syn_weight = types.MethodType(point_syn_weight, adaptor)

        return edge_adaptor


def point_syn_weight(self, edge, src_node, trg_node):
    return edge['syn_weight']*edge.nsyns
