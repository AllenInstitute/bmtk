import numpy as np
from pathlib import Path
import pickle as pkl
import pandas as pd
from pathlib import Path
import json
import tensorflow as tf
import hashlib
from time import time

from bmtk.utils import sonata
from .id_maps import TFIDMap
from .io_tools import io


def lex_sort_order_np(indices):
    if (
        len(indices)
        and np.issubdtype(indices.dtype, np.integer)
        and np.min(indices) >= 0
    ):
        second_max = int(indices[:, 1].max())
        stride = second_max + 1
        limit = np.iinfo(np.uint64).max
        if stride <= limit and int(indices[:, 0].max()) * stride + second_max <= limit:
            keys = indices[:, 0].astype(np.uint64) * np.uint64(stride)
            keys += indices[:, 1].astype(np.uint64, copy=False)
            return np.argsort(keys, kind="stable")
    return np.lexsort((indices[:, 1], indices[:, 0]))


def lex_sort_indices_np(indices, *arrays):
    sorted_ind = lex_sort_order_np(indices)
    sorted_arrays = list(map(lambda arr: arr[sorted_ind], [indices, *arrays]))
    return tuple(sorted_arrays)


def sort_indices_tf(indices, *arrays):
    indices = tf.cast(indices, dtype=tf.int64)
    max_ind = tf.reduce_max(indices) + 1
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = tf.argsort(q)
    indices = tf.cast(indices, dtype=tf.int32)
    sorted_arrays = [tf.gather(arr, sorted_ind).numpy() for arr in [indices, *arrays]]
    return tuple(sorted_arrays)


def compact_delays(delays):
    """Delays stored as uint8 or float16 when that is exact, else float32."""
    delays = np.asarray(delays, dtype=np.float32)
    for dtype in (np.uint8, np.float16):
        compact = delays.astype(dtype)
        if np.array_equal(compact.astype(np.float32), delays):
            return compact
    return delays


class NetworkAdaptor:
    def __init__(self, name, network_type, **options):
        self.name = name
        self.network_type = network_type
        self.componets_dir = None

        # For "input" networks, they must either support one type of input, ie "spikes" or
        # "current", or "na" (for unknown). Each input network can support multiple training
        # and predictions inputs but only of the same type.
        self._input_type = None
        self._input_options = {}
        self._source_tf_ids = None
        self._target_tf_ids = None
        self._edge_type_ids = None
        self._target_populations = set()
        self._source_populations = set()
        self.options = options

    @property
    def population_name(self):
        return self.name

    @property
    def n_nodes(self):
        raise NotImplementedError()

    @property
    def input_type(self):
        if self._input_type is None:
            # If it hasn't been set explicitly and is a "input" type of network, assume it's
            # network that is producing synaptic spikes
            if self.network_type == 'recurrent':
                return 'na'
            else:
                return 'spikes'
        else:
            return self._input_type

    @input_type.setter
    def input_type(self, _input_type):
        assert(_input_type in ['na', 'spikes', 'current'])
        if self._input_type is None:
            self._input_type = _input_type
        elif self._input_type != _input_type:
            # Due to the way the RNN is built, each input network can only support one type of input.
            # ie it can't both generate currents and spikes.
            raise ValueError(f'Attempting to set input for {self.name} network to {_input_type}, which has already been set to use {self._input_type} inputs.')

    @property
    def n_spiking_nodes(self):
        if self.input_type == 'spikes':
            return self.n_nodes
        else:
            return 0

    @property
    def input_options(self):
        return self._input_options

    @property
    def source_tf_ids(self):
        return self._source_tf_ids

    @property
    def target_tf_ids(self):
        return self._target_tf_ids

    @property
    def source_node_ids(self):
        tf2id_map = TFIDMap().tf2bmtk_id_map(populations=self._source_populations)
        return tf2id_map.loc[self.source_tf_ids]

    @property
    def target_node_ids(self):
        tf2id_map = TFIDMap().tf2bmtk_id_map(populations=self._target_populations)
        return tf2id_map.loc[self.target_tf_ids]

    @property
    def canonical_edge_order(self):
        return getattr(self, "_canonical_edge_order", None)

    @property
    def connection_table(self):
        # tf2id_map = TFIDMap().tf2bmtk_id_map(populations=self._target_populations)
        trg_ids = self.target_node_ids # tf2id_map.loc[self.target_tf_ids]
        src_ids = self.source_node_ids # tf2id_map.loc[self.source_tf_ids]

        return pd.DataFrame({
            'source_node_id': src_ids['node_id'].values,
            'source_population': src_ids['population'].values,
            'target_node_id': trg_ids['node_id'].values,
            'target_population': trg_ids['population'].values,
            'edge_type_id': self._edge_type_ids
        })

    def filter_ids(self, **query):
        raise NotImplementedError()

    def add_components_dirs(self, components_dirs=None):
        self.componets_dir = components_dirs

    def add_edges(self, edges_pop):
        pass

    def build(self, **opt_args):
        pass

    def get_bmtk_ids(self, **filter):
        raise NotImplementedError()

    def get_tf_ids(self, **filter):
        raise NotImplementedError()

    @staticmethod
    def merge(networks):
        return networks[0]

    @staticmethod
    def from_dict(networks_dict, allow_mixed_network=False):
        rec_networks = []
        input_networks = []
        names = set()

        # Reset shared syn_id mapping so all SONATA networks built here share
        # a single consistent set of syn_ids / basis_weights indices.
        SONATANetwork.reset_global_syn_id_mapping()

        if networks_dict.keys() == {'networks'}:
            networks_dict = networks_dict['networks']

        # These are network files that have been pre-processed in the (assumingly correct) layout and
        # stored as a pickle or npz file.
        for cached_dict in networks_dict.get('cached', []):
            network_type = cached_dict.pop('type')
            network_name = cached_dict.pop('name', None)
            file_format = cached_dict.pop('file_format', None)
            file_path = cached_dict.pop('file_path', None)

            io.log_debug(f'Loading network {network_name} from {file_path}')
            net = CachedNetwork(
                name=network_name, 
                network_type=network_type, 
                file_path=file_path, 
                file_format=file_format, 
                **cached_dict
            )

            # Make sure the same network is not accidentaly inserted twice.
            if net.name in names:
                raise ValueError(f'Found multiple networks with name "{network_name}.')
            else:
                names.add(network_name)

            if network_type == 'recurrent':
                rec_networks.append(net)
            elif network_type == 'input':
                input_networks.append(net)
            else:
                raise ValueError(f'Invalid "type" property value {network_type} in networks; valid options: "recurrent", "input"')

        for nodes in networks_dict.get('nodes', []):
            if not nodes.get('enabled', True):
                continue

            cache_file = nodes.get('cache_file', False)
            cache_check = nodes.get('cache_file', 'always')
            cache_overwrite = nodes.get('cache_overwrite', False)

            populations = nodes.get('population', None)
            populations = [populations] if isinstance(populations, (bytes, str)) else populations
            sonata_file = sonata.File(data_files=nodes['nodes_file'], data_type_files=nodes['node_types_file'])
            for node_pop in sonata_file.nodes.populations:
                node_pop_name = node_pop.name

                # If "populations" attribute defined make sure node_population is included in the list.
                if populations is not None and node_pop_name not in populations:
                    continue

                if node_pop_name in names:
                    raise ValueError(f'Found multiple networks with name "{network_name}.')
                else:
                    names.add(node_pop_name)

                # Need to check if a network contains input cells (virtual, filter), recurrent cells (point_neuron, biophysical, etc), or
                # a mixture of the two
                unique_model_types = set()
                for grp in node_pop.groups:
                    if "model_type" not in grp.all_columns:
                        raise ValueError(f'property "model_type" is missing from {node_pop_name} nodes files')
                    unique_model_types.update(set(np.unique(grp.get_values("model_type"))))

                has_virtual = 'virtual' in unique_model_types or 'filter' in unique_model_types
                has_non_virtual = unique_model_types - set(['virtual', 'filter']) != set()
                has_mixed_types = has_virtual and has_non_virtual

                io.log_debug(f'Loading network {node_pop_name} from {nodes["nodes_file"]}')
                if has_mixed_types:
                    if not allow_mixed_network:
                        raise ValueError(f'Nodes population {node_pop_name} from {nodes["nodes_file"]} has mix of virtual and non-virtual cells.')
                    else:
                        net, net_virt = SONATANetwork.split_up_virtual_nodes(node_pop)
                        rec_networks.append(SONATANetwork(net))
                        input_networks.append(SONATANetwork(net_virt))
                elif has_non_virtual:
                    rec_networks.append(SONATANetwork(node_pop, network_type='recurrent', cache_file=cache_file, cache_check=cache_check, cache_overwrite=cache_overwrite))
                elif has_virtual:
                    input_networks.append(SONATANetwork(node_pop, network_type='input', cache_file=cache_file, cache_check=cache_check, cache_overwrite=cache_overwrite))                   
                else:
                    # How did you get here!?!?
                    raise ValueError(f'Unable to resolve if {node_pop_name} has virtual or non-virtual cell')

        for edges in networks_dict.get('edges', []):
            if not edges.get('enabled', True):
                continue

            populations = edges.get('population', None)
            populations = [populations] if isinstance(populations, (bytes, str)) else populations
            sonata_file = sonata.File(data_files=edges['edges_file'], data_type_files=edges['edge_types_file'])
            for edge_pop in sonata_file.edges.populations:
                edge_pop_name = edge_pop.name

                src_pop = edge_pop.source_population
                trg_pop = edge_pop.target_population
                if not (src_pop in names and trg_pop in names):
                    raise ValueError(f'Orphaned edges {edge_pop_name} ({edges["edges_file"]}) does not have corresponding target and sources'
                                     f'populations {src_pop} -> {trg_pop}. Skipping.')

                for net in rec_networks + input_networks:
                    if net.name == src_pop:
                        io.log_debug(f'Adding edges to {net.name}')
                        net.add_edges(edge_pop)

        return rec_networks, input_networks

    @staticmethod
    def from_params(cache_file=None, network_type=None, network_name=None, file_format=None):
        rec_networks = []
        input_networks = []
        names = set()

        if cache_file is not None and Path(cache_file).is_file():
            # Make sure the same network is not accidentaly inserted twice.
            if network_name in names:
                raise ValueError(f'Found multiple networks with name "{network_name}.')
            else:
                names.add(network_name)

            if network_type not in ['recurrent', 'input']:
                raise ValueError(f'Invalid "type" property value {network_type} in networks; valid options: "recurrent", "input"')

            net = CachedNetwork(
                name=network_name, network_type=network_type, 
                file_path=cache_file, file_format=file_format
            )
            if network_type == 'recurrent':
                rec_networks.append(net)
            else:
                input_networks.append(net)

        return rec_networks, input_networks

    def get_nodes_df(self):
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError() 


class CachedNetwork(NetworkAdaptor):
    def __init__(self, name, file_path, network_type, file_format=None, **options):
        super().__init__(name=name, network_type=network_type, **options)
        self.file_path = file_path
        self.file_format = file_format or 'unknown'
        self.network_type = network_type
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f'Could not find network file {file_path}')
        
        self.network_dict = None
        if self.file_format in ['pickle', 'pkl', 'unknown']:
            try:
                with open(self.file_path, 'rb') as f:
                    self.network_dict = pkl.load(f)
            except Exception:
                pass

        if self.network_dict is None and self.file_format in ['numpy', 'npz', 'unknown']:
            try:
                with np.load(self.file_path) as data:
                    self.network_dict = data
            except Exception:
                pass

        if self.network_dict is None:
            raise ValueError(f'Could not load file "{self.file_path}", (format {self.file_format}) ')

        self.name = self.name or self.network_dict.get('name', None)
        if self.name is None:
            raise ValueError(f'Could not find unique name for network {self.file_path}. Pass in "name" option.')
       
        if self.network_type == 'recurrent':
            if 'tf_id_to_bmtk_id' in self.network_dict:
                node_ids = self.network_dict['tf_id_to_bmtk_id']
            else:
                node_ids = np.arange(self.network['n_nodes'], dtype=np.int64)
        elif self.network_type == 'input':
            node_ids = np.arange(self.network_dict['n_inputs'], dtype=np.int64)

        TFIDMap().add_bmtk_ids(
            population_name=self.population_name, 
            node_ids=node_ids,
            network_type=network_type
        )

    @property
    def n_nodes(self):
        return self.to_dict()['n_inputs']

    def to_dict(self):
        with open(self.file_path, 'rb') as f:
            network = pkl.load(f)
        network['options'] = self.options
        
        return network
       
    def get_nodes_df(self):
        raise NotImplementedError()

    @classmethod
    def load_cache(cls, file_path):
        try:
            adaptor = cls.load_pickle(file_path)
            return adaptor
        except Exception as e:
            pass

        raise ValueError(f'Could not load cache file {file_path}')

    @classmethod
    def load_pickle(cls, file_path):
        with open(file_path, 'rb') as f:
            network_dict = pkl.load(f)
        raise NotImplementedError()


class SONATANetwork(NetworkAdaptor):
    # Class-level shared mapping: dynamics_params_path -> global index.
    # Ensures all network instances (recurrent + inputs) use consistent syn_ids
    # that index into a single shared basis_weights table.
    _global_dyn_params_idx_lu = {}
    _global_ordered_dyn_param_dicts = []

    @classmethod
    def reset_global_syn_id_mapping(cls):
        """Reset the shared syn_id mapping (call before building a new model)."""
        cls._global_dyn_params_idx_lu = {}
        cls._global_ordered_dyn_param_dicts = []

    def __init__(self, sonata_node_pop, network_type, cache_file=False, filter=None, **opt_args):
        super().__init__(name=sonata_node_pop.name, network_type=network_type)
        self._sonata_node_pop = sonata_node_pop
        self._sonata_edge_pops = []
        self._dynamics_params_lu = {}
        self._cache_file = cache_file
        self._basis_weights = None
        self._generated_basis_weights = {}
        # self.population_name = sonata_node_pop.name

        self.id_maps = TFIDMap()
        self.id_maps.add_bmtk_ids(
            population_name=self.population_name, 
            node_ids=sonata_node_pop.node_ids,
            network_type=network_type
        )

        # for node in sonata_node_pop.filter(ei='i'):
        #     print(node.node_id)

    @property
    def md5hexsdigest(self):
        raise NotImplementedError()

    @property
    def n_nodes(self):
        return len(self._sonata_node_pop)

    def get_nodes_df(self):
        return self._sonata_node_pop.to_dataframe()

    def add_edges(self, sonata_edge_pop):
        self._sonata_edge_pops.append(sonata_edge_pop)
        self._source_populations.add(sonata_edge_pop.source_population)
        self._target_populations.add(sonata_edge_pop.target_population)

    def add_components_dirs(self, components_dirs=None):
        super().add_components_dirs(components_dirs=components_dirs)
        node_types_table = self._sonata_node_pop.types_table.to_dataframe()
        if 'dynamics_params' in node_types_table.columns:
            dyn_params_df = node_types_table.reset_index()[['node_type_id', 'model_type', 'dynamics_params']]

            def set_path(r):
                mt = r['model_type']
                json_path = r['dynamics_params']
                if Path(json_path).exists():
                    return json_path
                elif mt == 'biophysical':
                    params_dir = components_dirs['biophysical_neuron_models_dir']
                elif mt in ['point_process', 'point_neuron', 'point_soma']:
                    params_dir = components_dirs['point_neuron_models_dir']
                elif mt in ['population', 'rate_population']:
                    params_dir = components_dirs['population_models_dir']
                elif mt in ['lgnmodel', 'virtual', 'filter']:
                    params_dir = components_dirs['filter_models_dir']
                else:
                    params_dir = components_dirs['custom_neuron_models']
                return (Path(params_dir) / r['dynamics_params']).as_posix()

            def load_json(r):
                with open(r['full_path'], 'r') as f:
                    data = json.load(f)
                data['node_type_id'] = r['node_type_id']
                return data

            dyn_params_df.loc[:, 'full_path'] = dyn_params_df.apply(set_path, axis=1)
            self._dynamics_params_lu = pd.DataFrame(dyn_params_df.apply(load_json, axis=1).tolist()).to_dict(orient='list')

        # If required, update the "dynamics_params" path to include "synaptic_models_dir". eg pv2sst.json -> components/synanpatic_models/pv2sst.json
        # Wait until later to actual load the json files into memory, depending on the situation
        syn_models_dir = components_dirs.get('synaptic_models_dir', None)
        if syn_models_dir:
            syn_models_path = Path(syn_models_dir)
            for epop in self._sonata_edge_pops:
                etypes_table_df = epop.types_table
                # Edge-pop may not have an edge-types-table.csv file, or the file may not have "dynamics_params", or "dynamics_params"
                # may be an absolute path to the json file. In all such cases we don't preappend the synaptic_models_dir
                if etypes_table_df is None or 'dynamics_params' not in epop.types_table.columns:
                    continue

                for etype_id in np.unique(epop.type_ids):
                    dyn_params = etypes_table_df[etype_id]['dynamics_params']
                    if isinstance(dyn_params, str) and not Path(dyn_params).is_absolute():            
                        epop.types_table[etype_id]['dynamics_params'] = syn_models_path / dyn_params

        # To suppor the V1 model, allow users to pass in basis_weights_file, a csv file with synaptic params.
        basis_weights_file = components_dirs.get('basis_weights_file', None)
        if basis_weights_file:
            basis_weights_df = pd.read_csv(basis_weights_file)
            name_col = 'name' if 'name' in basis_weights_df.columns else 'connection_name'
            if name_col not in basis_weights_df.columns:
                raise ValueError(
                    f'basis_weights_file {basis_weights_file} must contain either a "name" or '
                    '"connection_name" column.'
                )
            weight_columns = sorted(
                [
                    column
                    for column in basis_weights_df
                    if column.startswith("w") and column[1:].isdigit()
                ],
                key=lambda column: int(column[1:]),
            )
            if not weight_columns or weight_columns != [
                f"w{index}" for index in range(len(weight_columns))
            ]:
                raise ValueError(
                    "basis_weights_file must contain consecutive w0, w1, ... columns"
                )
            self._basis_weights = {
                r[name_col]: r[weight_columns].to_numpy(dtype=float)
                for _, r in basis_weights_df.iterrows()
            }

    def node_attribute_values(self, columns):
        """Per-neuron values of node attributes, in TF order (the node file order).

        Columns of the node-types table are resolved per node type, so whole-brain
        populations need no per-node dataframe; other columns are read from a
        single-group population's node group.
        """
        pop = self._sonata_node_pop
        types = pop.types_table.to_dataframe()
        type_ids = pop.type_ids
        values, group = {}, None
        for column in columns:
            if column in types.columns:
                values[column] = types[column].reindex(type_ids).to_numpy()
                continue
            if len(pop.group_ids) != 1:
                raise ValueError(
                    f'Node attribute {column!r} is not a node-types column and population '
                    f'{pop.name!r} has several node groups.'
                )
            if group is None:
                group = pop.get_group(pop.group_ids[0]).to_dataframe()
            if column not in group.columns:
                raise ValueError(f'Node attribute {column!r} not found in population {pop.name!r}.')
            values[column] = group[column].to_numpy()[np.asarray(pop.group_index_ds)]
        return values

    def get_bmtk_ids(self, **filter):
        node_ids = {self._sonata_node_pop.name: []}
        for n in self._sonata_node_pop.filter(**filter):
            node_ids[self._sonata_node_pop.name].append(n.node_id)

        return node_ids

    def get_tf_ids(self, **filter):
        bmtk2tf_id_map = TFIDMap().recurrent_bmtk_ids()
        node_ids_dict = self.get_bmtk_ids(**filter)
        tf_ids = np.empty(0, dtype=int)
        for pop_name, pop_node_ids in node_ids_dict.items():
            _pop_tf_ids = bmtk2tf_id_map[pop_name][pop_node_ids]
            tf_ids = np.concatenate((tf_ids, _pop_tf_ids))
        return tf_ids

    def _synaptic_dyn_params(self, format='dict'):
        # Use class-level shared mapping to ensure all networks (recurrent + inputs)
        # assign consistent syn_ids that index into one global basis_weights table.
        dynamic_params_idx_lu = SONATANetwork._global_dyn_params_idx_lu
        ordered_dyn_param_dicts = SONATANetwork._global_ordered_dyn_param_dicts

        data_table = {}
        for edge_pop in self._sonata_edge_pops:
            _data_table = {
                'edge_type_ids': [],
                'param_idx': [],
                'param_paths': []
            }

            edge_types_table = edge_pop.types_table
            for etid in np.unique(edge_pop.type_ids):
                dyn_params_path = edge_types_table[etid]['dynamics_params']
                dyn_params_idx = dynamic_params_idx_lu.get(dyn_params_path, None)
                # dyn_params_dict = dynamic_params_lu.get(dyn_params_path, None)
                if dyn_params_idx is None:
                    with open(dyn_params_path, 'r') as f:
                        dyn_params_dict = json.load(f)

                    generated_weights = getattr(
                        self, "_generated_basis_weights", {}
                    ).get(Path(dyn_params_path))
                    if generated_weights is not None:
                        dyn_params_dict["basis_weights"] = generated_weights
                    elif self._basis_weights is not None:
                        basis_name = Path(dyn_params_path).stem
                        if basis_name in self._basis_weights:
                            dyn_params_dict['basis_weights'] = self._basis_weights[basis_name]

                    dyn_params_idx = len(ordered_dyn_param_dicts)
                    dynamic_params_idx_lu[dyn_params_path] = dyn_params_idx
                    ordered_dyn_param_dicts.append(dyn_params_dict)

                _data_table['edge_type_ids'].append(etid)
                _data_table['param_idx'].append(dyn_params_idx)
                _data_table['param_paths'].append(Path(dyn_params_path).name)
                data_table[edge_pop.name] = _data_table

        syn_params_lu = {}
        for k, v in data_table.items():
            lu_df = pd.DataFrame({
                'edge_type_ids': v['edge_type_ids'],
                'dyn_params_idx': v['param_idx'],
                'dyn_params_path': v['param_paths'],
            }).set_index('edge_type_ids')
            syn_params_lu[k] = lu_df

        if format == 'list':
            # return synaptic dynamics params as a list of dictionaries
            dyn_params_ret = ordered_dyn_param_dicts
        if format == 'dict':
            # return synaptic dynamics params as a dictionary of lists
            dyn_params_ret = pd.DataFrame(ordered_dyn_param_dicts).to_dict(orient='list')
        else:
            raise ValueError() 

        return dyn_params_ret, syn_params_lu

    @staticmethod
    def split_up_virtual_nodes(sonata_node_pop):
        raise NotImplementedError()

    def _build_recurrent_dict(self):
        # if self._cache_file and Path(self._cache_file).exists():
        #     with open(self._cache_file, 'rb') as f:
        #         return pkl.load(f)

        # dict = super().to_dict()
        n_nodes = len(self._sonata_node_pop.node_ids)

        n_edges = 0
        for edge_pop in self._sonata_edge_pops:
            n_edges += len(edge_pop)

        self._source_tf_ids = np.zeros((n_edges,), dtype=np.uint32)
        self._target_tf_ids = np.zeros(n_edges, dtype=np.uint32)
        self._edge_type_ids = np.zeros(n_edges, dtype=np.uint32)
        weights = np.zeros(n_edges, dtype=np.float32)
        delays = np.zeros(n_edges, dtype=np.float32)
        syn_ids = np.zeros(n_edges, dtype=np.uint8)
        syn_dyn_params, syn_dyn_parms_lu = self._synaptic_dyn_params()

        population_offset = 0
        for edge_pop in self._sonata_edge_pops:
            et_table = edge_pop.types_table.to_dataframe()
            src_pop = edge_pop.source_population 
            trg_pop = edge_pop.target_population

            for model_grp in edge_pop.groups:
                rows, group_indices = self._edge_group_rows(edge_pop, model_grp)
                destination = rows + population_offset
                self._source_tf_ids[destination] = self.id_maps.bmtk2tf_id_map(src_pop)[
                    model_grp.src_node_ids[()]
                ]
                self._target_tf_ids[destination] = self.id_maps.bmtk2tf_id_map(trg_pop)[
                    model_grp.trg_node_ids[()]
                ]
                self._edge_type_ids[destination] = model_grp.edge_type_ids

                if 'syn_weight' in model_grp.columns:
                    weights[destination] = model_grp.get_values(
                        "syn_weight", all_rows=True
                    )[group_indices]
                elif 'syn_weight' in et_table.columns:
                    weights[destination] = et_table.loc[model_grp.edge_type_ids][
                        "syn_weight"
                    ]
                else:
                    raise Exception()

                if 'delay' in model_grp.columns:
                    delays[destination] = model_grp.get_values("delay", all_rows=True)[
                        group_indices
                    ]
                elif 'delay' in et_table.columns:
                    delays[destination] = et_table.loc[model_grp.edge_type_ids]["delay"]
                else:
                    raise Exception()

                if 'dynamics_params' in model_grp.columns:
                    raise NotImplementedError()
                elif 'dynamics_params' in et_table.columns:
                    syn_ids[destination] = (
                        syn_dyn_parms_lu[edge_pop.name]
                        .loc[model_grp.edge_type_ids]["dyn_params_idx"]
                        .values
                    )
                else:
                    raise Exception()
            population_offset += len(edge_pop)

        indices = np.column_stack((self._target_tf_ids, self._source_tf_ids))
        sort_order = lex_sort_order_np(indices)
        self._canonical_edge_order = np.empty(len(sort_order), dtype=sort_order.dtype)
        self._canonical_edge_order[sort_order] = np.arange(len(sort_order))
        indices = indices[sort_order]
        weights = weights[sort_order]
        delays = delays[sort_order]
        syn_ids = syn_ids[sort_order]
        self._edge_type_ids = self._edge_type_ids[sort_order]
        self._target_tf_ids = self._target_tf_ids[sort_order]
        self._source_tf_ids = self._source_tf_ids[sort_order]

        rec_dict = {
            'name': self.population_name,
            'network_type': self.network_type,
            'n_nodes': n_nodes,
            'node_type_ids': self._recurrent_node_type_ids(),
            'node_params': self._recurrent_node_params(),
            'synapses': {
                'indices': indices,
                'weights': weights.astype(np.float32),
                'delays': delays,
                'dense_shape': (n_nodes, n_nodes),
                'syn_ids': syn_ids,
                'dynamics_params': syn_dyn_params
            }
        }

        if self._cache_file and not Path(self._cache_file).exists():
            Path(self._cache_file).parent.mkdir(exist_ok=True, parents=True)
            with open(self._cache_file, 'wb') as f:
                pkl.dump(rec_dict, f)

        return rec_dict

    def _recurrent_node_type_ids(self):
        """Per-neuron index into the node-type parameter tables, in TF order."""
        ntids_lu = pd.DataFrame({
            'node_type_ids': self._dynamics_params_lu['node_type_id'],
            'order': range(len(self._dynamics_params_lu['node_type_id']))
        }).set_index('node_type_ids')
        return np.array(ntids_lu.loc[self._sonata_node_pop.type_ids], dtype=np.int32).flatten()

    def _recurrent_node_params(self):
        """Per-node-type GLIF parameter tables."""
        lu = self._dynamics_params_lu
        params = {
            'V_th': np.array(lu['V_th'], dtype=np.float32),
            'g': np.array(lu['g'], dtype=np.float32),
            'E_L': np.array(lu['E_L'], dtype=np.float32),
            'k': np.array(lu['asc_decay'], dtype=np.float32),
            'C_m': np.array(lu['C_m'], dtype=np.float32),
            'V_reset': np.array(lu['V_reset'], dtype=np.float32),
            't_ref': np.array(lu['t_ref'], dtype=np.float32),
            'asc_amps': np.array(lu['asc_amps'], dtype=np.float32)
        }
        for name, default in (
            ("V_m", params["E_L"]),
            ("asc_init", np.zeros_like(params["k"])),
            ("asc_r", np.ones_like(params["k"])),
        ):
            params[name] = np.asarray(lu[name], dtype=np.float32) if name in lu else default
        return params

    def node_metadata_dict(self):
        """Whole-network node fields of the recurrent dict, without reading any edge."""
        return {
            'name': self.population_name,
            'network_type': self.network_type,
            'n_nodes': self.n_nodes,
            'node_type_ids': self._recurrent_node_type_ids(),
            'node_params': self._recurrent_node_params(),
        }

    def partition_recurrent_dicts(self, partition, target_offset, input_options):
        """``(recurrent_dict, partition_input_dict, canonical_ids)`` of this rank's neurons.

        Edges into the rank's neurons are read in chunks and ordered as the
        global canonical order (target, source, file row); those from local
        neurons form the rank's recurrence, those from other ranks' neurons the
        partition input (one column per remote source neuron, ascending).
        ``canonical_ids`` maps each master (``"<recurrent>"`` and the partition
        input) to the global canonical edge index of its edges:
        ``target_offset[target]`` plus the edge's position among its target's
        edges, all of which live on the target's owner.
        """
        from .neuron_partition import PARTITION_INPUT

        syn_dyn_params, _ = self._synaptic_dyn_params()
        edges = self.read_edges(keep_target=partition.owner == partition.rank)
        order = np.lexsort((edges["rows"], edges["source"], edges["target"]))
        edges = {name: values[order] for name, values in edges.items()}
        target, source = edges["target"], edges["source"]
        first = np.r_[0, np.flatnonzero(np.diff(target)) + 1]
        within = np.arange(target.size) - np.repeat(first, np.diff(np.r_[first, target.size]))
        canonical = np.asarray(target_offset, dtype=np.int64)[target] + within
        local_source = partition.owner[source] == partition.rank
        remote = ~local_source
        remote_sources = np.unique(source[remote])
        placeholder = remote_sources.size == 0
        if placeholder:
            # No edges from other ranks: one edgeless placeholder column keeps this rank in
            # every spike/credit exchange its peers need.
            remote_sources = np.array([np.flatnonzero(partition.owner != partition.rank)[0]])
        position = partition.position
        n_local = partition.n_local
        rec_dict = {
            'name': self.population_name,
            'network_type': self.network_type,
            'n_nodes': n_local,
            'node_type_ids': self._recurrent_node_type_ids()[partition.local_nodes],
            'node_params': self._recurrent_node_params(),
            'synapses': {
                'indices': np.column_stack((position[target[local_source]], position[source[local_source]])),
                'weights': edges["weights"][local_source],
                'delays': edges["delays"][local_source],
                'dense_shape': (n_local, n_local),
                'syn_ids': edges["syn_ids"][local_source],
                'dynamics_params': syn_dyn_params,
            },
            'tf_id_to_global': partition.local_nodes,
        }
        input_dict = {
            'name': PARTITION_INPUT,
            'network_type': 'input',
            'input_type': 'spikes',
            'node_params': {},
            'n_inputs': int(remote_sources.size),
            'indices': np.column_stack(
                (position[target[remote]], np.searchsorted(remote_sources, source[remote]))
            ),
            'weights': edges["weights"][remote],
            'delays': edges["delays"][remote],
            'syn_ids': edges["syn_ids"][remote],
            'options': dict(input_options),
            'remote_sources': remote_sources,
            'partition_placeholder': placeholder,
        }
        canonical_ids = {'<recurrent>': canonical[local_source], PARTITION_INPUT: canonical[remote]}
        # Global (target, source), edge type and file delay of every master edge in its
        # canonical order, kept compact for weight-group keys and per-rank export.
        index_dtype = np.int32 if self.n_nodes < np.iinfo(np.int32).max else np.int64
        type_dtype = np.uint16 if edges["edge_type_ids"].max(initial=0) <= np.iinfo(np.uint16).max else np.uint32
        self._partition_endpoints, self._partition_edge_types, self._partition_delays = {}, {}, {}
        for name, mask in (('<recurrent>', local_source), (PARTITION_INPUT, remote)):
            self._partition_endpoints[name] = (target[mask].astype(index_dtype), source[mask].astype(index_dtype))
            self._partition_edge_types[name] = edges["edge_type_ids"][mask].astype(type_dtype)
            self._partition_delays[name] = compact_delays(edges["delays"][mask])
        return rec_dict, input_dict, canonical_ids

    def prepare_export_tables(self):
        """Whole-network edge tables for weight export when this rank built only a partition.

        Sets the canonical-ordered (recurrent) or file-ordered (input) endpoint and
        edge-type arrays and the canonical order that :class:`ModelWeights` uses.
        """
        if getattr(self, "_export_tables_ready", False):
            return
        edges = self.read_edges()
        if self.network_type == 'recurrent':
            order = np.lexsort((edges["rows"], edges["source"], edges["target"]))
            self._canonical_edge_order = np.empty(len(order), dtype=order.dtype)
            self._canonical_edge_order[order] = np.arange(len(order))
        else:
            order = np.arange(len(edges["rows"]))
        self._source_tf_ids = edges["source"][order].astype(np.uint32)
        self._target_tf_ids = edges["target"][order].astype(np.uint32)
        self._edge_type_ids = edges["edge_type_ids"][order]
        self._export_tables_ready = True

    def partition_input_dict(self, partition):
        """``(input_dict, global_rows)``: this input's edges into the rank's neurons, file order."""
        edges = self.read_edges(keep_target=partition.owner == partition.rank)
        net_dict = {
            'name': self.name,
            'network_type': self.network_type,
            'input_type': self.input_type,
            'node_params': self._sonata_node_pop.to_dataframe().to_dict('list'),
            'n_inputs': len(self._sonata_node_pop.node_ids),
            'indices': np.column_stack((partition.position[edges["target"]], edges["source"])),
            'weights': edges["weights"],
            'delays': edges["delays"],
            'syn_ids': edges["syn_ids"],
            'options': dict(self.input_options),
        }
        net_dict['options'] = self.options
        return net_dict, edges["rows"]

    @staticmethod
    def _edge_group_rows(edge_pop, model_grp):
        count, ranges = edge_pop.group_indicies(model_grp.group_id)
        rows = (
            np.concatenate(
                [np.arange(start, stop, dtype=np.int64) for start, stop in ranges]
            )
            if count
            else np.empty(0, dtype=np.int64)
        )
        group_indices = (
            np.concatenate(
                [edge_pop._group_index_ds[start:stop] for start, stop in ranges]
            ).astype(np.int64, copy=False)
            if count
            else np.empty(0, dtype=np.int64)
        )
        return rows, group_indices

    # -- rank-local (neuron-partitioned) edge reading ------------------------------------

    _EDGE_CHUNK_ROWS = 1 << 22

    def _iter_edge_chunks(self, row_slice=None):
        """Yield ``(edge_pop, population_offset, start, stop)`` row chunks of every edge population.

        ``row_slice=(index, count)`` restricts to that share of each population's
        rows (used to split one scan across ranks).
        """
        population_offset = 0
        for edge_pop in self._sonata_edge_pops:
            n_rows = len(edge_pop)
            begin, end = 0, n_rows
            if row_slice is not None:
                index, count = row_slice
                begin, end = n_rows * index // count, n_rows * (index + 1) // count
            for start in range(begin, end, self._EDGE_CHUNK_ROWS):
                yield edge_pop, population_offset, start, min(start + self._EDGE_CHUNK_ROWS, end)
            population_offset += n_rows

    @staticmethod
    def _read_dataset_rows(dataset, rows, span):
        """``dataset[rows]`` (any order, repeats allowed) through windows of at most ``span`` rows."""
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        result = np.empty(rows.size, dtype=dataset.dtype)
        start = 0
        while start < sorted_rows.size:
            low = int(sorted_rows[start])
            stop = int(np.searchsorted(sorted_rows, low + span, side="left"))
            block = np.asarray(dataset[low:int(sorted_rows[stop - 1]) + 1])
            result[order[start:stop]] = block[sorted_rows[start:stop] - low]
            start = stop
        return result

    def target_in_degree(self, row_slice=None):
        """Edges per target TF id (int64), optionally over a row share only."""
        counts = np.zeros(self.n_nodes, dtype=np.int64)
        for edge_pop, _, start, stop in self._iter_edge_chunks(row_slice):
            targets = self.id_maps.bmtk2tf_id_map(edge_pop.target_population)[
                edge_pop._target_node_id_ds[start:stop]
            ]
            counts += np.bincount(np.asarray(targets, dtype=np.int64), minlength=self.n_nodes)
        return counts

    def read_edges(self, keep_target=None):
        """Edges whose target TF id satisfies ``keep_target`` (boolean array by TF id; ``None``: all).

        Returns a dict of arrays in global row order: ``rows`` (global row over the
        concatenated edge populations), ``source``/``target`` TF ids,
        ``edge_type_ids``, ``weights``, ``delays`` and ``syn_ids``, with the values
        :meth:`_build_recurrent_dict` / :meth:`_build_input_dict` read. Rows are
        read in chunks, so only kept edges are held.
        """
        _, syn_dyn_parms_lu = self._synaptic_dyn_params()
        parts = {name: [] for name in ("rows", "source", "target", "edge_type_ids", "weights", "delays", "syn_ids")}
        groups_by_population = {}
        for edge_pop, offset, start, stop in self._iter_edge_chunks():
            key = id(edge_pop)
            if key not in groups_by_population:
                groups_by_population[key] = (
                    {group.group_id: group for group in edge_pop.groups},
                    edge_pop.types_table.to_dataframe(),
                )
            groups, et_table = groups_by_population[key]
            targets = np.asarray(
                self.id_maps.bmtk2tf_id_map(edge_pop.target_population)[edge_pop._target_node_id_ds[start:stop]]
            )
            keep = np.ones(targets.size, dtype=bool) if keep_target is None else keep_target[targets]
            if not keep.any():
                continue
            selected = np.flatnonzero(keep)
            sources = np.asarray(
                self.id_maps.bmtk2tf_id_map(edge_pop.source_population)[edge_pop._source_node_id_ds[start:stop]]
            )[selected]
            edge_types = np.asarray(edge_pop._type_id_ds[start:stop])[selected]
            group_ids = np.asarray(edge_pop._group_id_ds[start:stop])[selected]
            group_rows = np.asarray(edge_pop._group_index_ds[start:stop])[selected]
            weights = np.empty(selected.size, dtype=np.float32)
            delays = np.empty(selected.size, dtype=np.float32)
            for group_id in np.unique(group_ids):
                in_group = group_ids == group_id
                group = groups[group_id]
                rows = group_rows[in_group]
                types = edge_types[in_group]
                for column, destination in (("syn_weight", weights), ("delay", delays)):
                    if column in group.columns:
                        destination[in_group] = self._read_dataset_rows(
                            group._h5_group[column], rows, self._EDGE_CHUNK_ROWS,
                        )
                    elif column in et_table.columns:
                        destination[in_group] = et_table.loc[types][column].to_numpy()
                    else:
                        raise ValueError(f"Edges of {edge_pop.name} have no {column!r} values.")
                if "dynamics_params" in group.columns:
                    raise NotImplementedError()
            if "dynamics_params" not in et_table.columns:
                raise ValueError(f"Edges of {edge_pop.name} have no dynamics_params.")
            parts["rows"].append(offset + start + selected)
            parts["source"].append(sources.astype(np.int64))
            parts["target"].append(targets[selected].astype(np.int64))
            parts["edge_type_ids"].append(edge_types.astype(np.uint32))
            parts["weights"].append(weights)
            parts["delays"].append(delays)
            parts["syn_ids"].append(
                syn_dyn_parms_lu[edge_pop.name].loc[edge_types]["dyn_params_idx"].to_numpy().astype(np.uint8)
            )
        empty = {"rows": np.int64, "source": np.int64, "target": np.int64, "edge_type_ids": np.uint32,
                 "weights": np.float32, "delays": np.float32, "syn_ids": np.uint8}
        return {
            name: np.concatenate(values) if values else np.empty(0, dtype=empty[name])
            for name, values in parts.items()
        }

    def _build_input_dict(self):
        # if self._cache_file and Path(self._cache_file).exists():
        #     with open(self._cache_file, 'rb') as f:
        #         return pkl.load(f)

        edge_pop = self._sonata_edge_pops[0]
        et_table = edge_pop.types_table.to_dataframe()

        n_edges = 0
        for edge_pop in self._sonata_edge_pops:
            n_edges += len(edge_pop)

        self._source_tf_ids = np.zeros((n_edges,), dtype=np.uint32)
        self._target_tf_ids = np.zeros(n_edges, dtype=np.uint32)
        self._edge_type_ids = np.zeros(n_edges, dtype=np.uint32)
        weights = np.zeros(n_edges, dtype=np.float32)
        delays = np.zeros(n_edges, dtype=np.float32)
        syn_ids = np.zeros(n_edges, dtype=np.uint8)

        population_offset = 0
        syn_dyn_params, syn_dyn_parms_lu = self._synaptic_dyn_params()
        for edge_pop in self._sonata_edge_pops:
            et_table = edge_pop.types_table.to_dataframe()
            src_pop = edge_pop.source_population 
            trg_pop = edge_pop.target_population

            for model_grp in edge_pop.groups:
                rows, group_indices = self._edge_group_rows(edge_pop, model_grp)
                destination = rows + population_offset
                self._source_tf_ids[destination] = self.id_maps.bmtk2tf_id_map(src_pop)[
                    model_grp.src_node_ids[()]
                ]
                self._target_tf_ids[destination] = self.id_maps.bmtk2tf_id_map(trg_pop)[
                    model_grp.trg_node_ids[()]
                ]
                self._edge_type_ids[destination] = model_grp.edge_type_ids

                if 'syn_weight' in model_grp.columns:
                    weights[destination] = model_grp.get_values(
                        "syn_weight", all_rows=True
                    )[group_indices]
                elif 'syn_weight' in et_table.columns:
                    weights[destination] = et_table.loc[model_grp.edge_type_ids][
                        "syn_weight"
                    ]
                else:
                    raise Exception()

                if 'delay' in model_grp.columns:
                    delays[destination] = model_grp.get_values("delay", all_rows=True)[
                        group_indices
                    ]
                elif 'delay' in et_table.columns:
                    delays[destination] = et_table.loc[model_grp.edge_type_ids]["delay"]
                else:
                    raise Exception() 

                if 'dynamics_params' in model_grp.columns:
                    raise NotImplementedError()
                elif 'dynamics_params' in et_table.columns:
                    syn_ids[destination] = (
                        syn_dyn_parms_lu[edge_pop.name]
                        .loc[model_grp.edge_type_ids]["dyn_params_idx"]
                        .values
                    )
                else:
                    raise Exception()
            population_offset += len(edge_pop)

        node_params = self._sonata_node_pop.to_dataframe().to_dict('list')
        net_dict = {
            'name': self.name,
            'network_type': self.network_type,
            'input_type': self.input_type,
            'node_params': node_params,
            'n_inputs': len(self._sonata_node_pop.node_ids),
            'indices': np.column_stack((self._target_tf_ids, self._source_tf_ids)),
            'weights': weights,
            'delays': delays,
            'syn_ids': syn_ids,
            'options': dict(self.input_options)
        }

        if self._cache_file and not Path(self._cache_file).exists():
            Path(self._cache_file).parent.mkdir(exist_ok=True, parents=True)
            with open(self._cache_file, 'wb') as f:
                pkl.dump(net_dict, f)

        return net_dict

    def to_dict(self):
        if self.network_type == 'recurrent':
            net_dict = self._build_recurrent_dict()
            # return self._build_recurrent_dict()
        elif self.network_type == 'input':
            net_dict = self._build_input_dict()
            # return self._build_input_dict()
        else:
            raise ValueError(f'Unknown network_type {self.network_type}')

        net_dict['options'] = self.options
        return net_dict
