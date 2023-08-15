import numpy as np
from collections import Counter
import numbers
import nest
import types
import pandas as pd

from bmtk.simulator.core.sonata_reader import NodeAdaptor, SonataBaseNode, EdgeAdaptor, SonataBaseEdge
from bmtk.simulator.pointnet.io_tools import io
from bmtk.simulator.pointnet.pyfunction_cache import py_modules
from bmtk.simulator.pointnet.glif_utils import convert_aibs2nest
from bmtk.simulator.pointnet.nest_utils import nest_version

NEST_SYNAPSE_MODEL_PROP = 'model' if nest_version[0] == 2 else 'synapse_model'



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
        self._nest_objs = []
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
        self._nest_objs = nest.Create(self.nest_model, self.n_nodes, self.nest_params)
        self._nest_ids = self._nest_objs.tolist() if nest_version[0] >= 3 else self._nest_objs


class PointNode(SonataBaseNode):
    def __init__(self, node, prop_adaptor):
        super(PointNode, self).__init__(node, prop_adaptor)
        self._nest_objs = []
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
            self._nest_objs = nest.Create(nest_model, 1, dynamics_params)
        else:
            cell_fnc = py_modules.cell_processor(fnc_name)
            self._nest_objs = cell_fnc(nest_model, self._node, dynamics_params)

        self._nest_ids = self._nest_objs.tolist() if nest_version[0] >= 3 else self._nest_objs


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
    def preprocess_node_types(network, node_population):
        NodeAdaptor.preprocess_node_types(network, node_population)
        node_types_table = node_population.types_table
        if 'model_template' in node_types_table.columns and 'dynamics_params' in node_types_table.columns:
            node_type_ids = np.unique(node_population.type_ids)
            for nt_id in node_type_ids:
                node_type_attrs = node_types_table[nt_id]
                mtemplate = node_type_attrs['model_template']
                dyn_params = node_type_attrs['dynamics_params']
                if mtemplate.startswith('nest:glif') and dyn_params.get('type', None) == 'GLIF':
                    model_template, dynamics_params = convert_aibs2nest(mtemplate, dyn_params)
                    node_type_attrs['model_template'] = model_template
                    node_type_attrs['dynamics_params'] = dynamics_params

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
            io.log_info('Batch processing nodes for {}/{}.'.format(node_group.parent.name, node_group.group_id))

        return node_adaptor


class PointEdge(SonataBaseEdge):
    @property
    def source_node_ids(self):
        return [self._edge.source_node_id]

    @property
    def target_node_ids(self):
        return [self._edge.target_node_id]

    @property
    def nest_params(self):
        if self.model_template in py_modules.synapse_models:
            src_node = self._prop_adaptor._network.get_node_id(self.source_population, self.source_node_id)
            trg_node = self._prop_adaptor._network.get_node_id(self.target_population, self.target_node_id)
            syn_model_fnc = py_modules.synapse_model(self.model_template)
        else:
            src_node = None
            trg_node = None
            syn_model_fnc = py_modules.synapse_models('default')

        return syn_model_fnc(self, src_node, trg_node)


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
        # TODO: THIS NEEDS to be replaced with call to synapse_models
        params_dict = {'weight': self.syn_weight(edge, None, None), 'delay': edge.delay}
        params_dict.update(edge.dynamics_params)
        return params_dict

    def get_edge(self, sonata_node):
        return PointEdge(sonata_node, self)

    @staticmethod
    def preprocess_edge_types(network, edge_population):
        # Fix for sonata/300_pointneurons
        EdgeAdaptor.preprocess_edge_types(network, edge_population)
        edge_types_table = edge_population.types_table
        edge_type_ids = np.unique(edge_population.type_ids)

        for et_id in edge_type_ids:
            edge_type = edge_types_table[et_id]
            if 'model_template' in edge_types_table.columns:
                model_template = edge_type['model_template']
                if model_template.startswith('nest'):
                    edge_type['model_template'] = model_template[5:]

    def get_batches(self, edge_group):
        src_ids = {}
        trg_ids = {}

        edge_types_table = edge_group.parent.edge_types_table

        edge_type_ids = edge_group.edge_type_ids
        et_id_counter = Counter(edge_type_ids)
        tmp_df = pd.DataFrame({'etid': edge_type_ids, 'src_nids': edge_group.src_node_ids,
                               'trg_nids': edge_group.trg_node_ids})

        if 'nsyns' in edge_group.columns:
            tmp_df['nsyns'] = edge_group.get_dataset('nsyns')

        if 'syn_weight' in edge_group.columns:
            tmp_df['syn_weight'] = edge_group.get_dataset('syn_weight')

        if 'delay' in edge_group.columns:
            tmp_df['delay'] = edge_group.get_dataset('delay')

        #for et_id, grp_vals in tmp_df.groupby('etid'):
        #    src_ids[et_id] = np.array(grp_vals['src_nids'])
        #    trg_ids[et_id] = np.array(grp_vals['trg_nids'])

        type_params = {edge_id: {} for edge_id in et_id_counter.keys()}
        src_pop = edge_group.parent.source_population
        trg_pop = edge_group.parent.target_population

        grp_df = None
        src_nodes_df = None
        trg_nodes_df = None
        for edge_id, grp_vals in tmp_df.groupby('etid'):
            edge_props = edge_types_table[edge_id]
            n_edges = len(grp_vals)

            # Get the model type
            type_params[edge_id][NEST_SYNAPSE_MODEL_PROP] = edge_props['model_template']

            # Add dynamics params
            # TODO: Add to dataframe and if a part of hdf5 we can return any dynamics params as a list
            type_params[edge_id].update(edge_props['dynamics_params'])

            # get the delay parameter
            if 'delay' in grp_vals.columns:
                type_params[edge_id]['delay'] = grp_vals['delay']
            elif 'delay' in edge_props.keys():
                delay = edge_props['delay']
                # For NEST 2.* 'delay' can be a single value, but for 3.* it requires a full array for each edge
                delay = np.full(n_edges, delay) if nest_version[0] >= 3 else delay
                type_params[edge_id]['delay'] = delay

            weight_function = edge_types_table[edge_id].get('weight_function', None)
            if weight_function is not None:
                if grp_df is None:
                    grp_df = edge_group.to_dataframe()
                    src_nodes_df = self._network.get_nodes_df(src_pop)
                    trg_nodes_df = self._network.get_nodes_df(trg_pop)
                edges = grp_df[grp_df['edge_type_id'] == edge_id]
                target_nodes = trg_nodes_df.loc[edges['target_node_id'].values]
                source_nodes = src_nodes_df.loc[edges['source_node_id'].values]

                if not py_modules.has_synaptic_weight(weight_function):
                    err_msg = 'Unable to calculate synaptic weight for "{}" edges, missing "weight_function" ' \
                              'attribute value {} function.'.format(edge_group.parent.name, weight_function)
                    io.log_exception(err_msg)

                weight_fnc = py_modules.synaptic_weight(weight_function)
                type_params[edge_id]['weight'] = weight_fnc(edges, source_nodes, target_nodes)

            else:
                # Get nsyns as either an array or a constant. If not explcitiy specified assume nsyns = 1
                if 'nsyns' in grp_vals.columns:
                    nsyns = grp_vals['nsyns'].values
                else:
                    nsyns = edge_props.get('nsyns', 1)

                # get syn_weight as either an array or constant. If not explicity stated throw an error
                if 'syn_weight' in grp_vals.columns:
                    syn_weight = grp_vals['syn_weight'].values
                elif 'syn_weight' in edge_props.keys():
                    syn_weight = edge_props['syn_weight']
                else:
                    # TODO: Make more explicity. Or default to syn_weight of 0
                    raise Exception('Could not find syn_weight value')

                # caluclate weight
                type_params[edge_id]['weight'] = nsyns * syn_weight

            yield PointEdgeBatched(source_nids=grp_vals['src_nids'].values, target_nids=grp_vals['trg_nids'].values,
                                   nest_params=type_params[edge_id])

    @staticmethod
    def patch_adaptor(adaptor, edge_group):
        edge_adaptor = EdgeAdaptor.patch_adaptor(adaptor, edge_group)

        if 'weight_function' not in edge_group.all_columns and 'syn_weight' in edge_group.all_columns:
            adaptor.syn_weight = types.MethodType(point_syn_weight, adaptor)

        #else:
        #    edge_adaptor.batch_process = False

        return edge_adaptor


def point_syn_weight(self, edge, src_node, trg_node):
    return edge['syn_weight']*edge.nsyns
