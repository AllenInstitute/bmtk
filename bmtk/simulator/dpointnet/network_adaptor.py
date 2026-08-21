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
            self._basis_weights = {r[name_col]: np.array([r['w0'], r['w1'], r['w2'], r['w3']]) for _, r in basis_weights_df.iterrows()}

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
                    
                    if self._basis_weights is not None:
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

        ntids_lu = pd.DataFrame({
            'node_type_ids': self._dynamics_params_lu['node_type_id'],
            'order': range(len(self._dynamics_params_lu['node_type_id']))
        }).set_index('node_type_ids')

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
        
        idx_beg, idx_end = 0, 0
        for edge_pop in self._sonata_edge_pops:
            et_table = edge_pop.types_table.to_dataframe()
            idx_beg = idx_end
            idx_end = idx_beg + len(edge_pop)
            src_pop = edge_pop.source_population 
            trg_pop = edge_pop.target_population
            
            idx_beg = 0
            for model_grp in edge_pop.groups:
                idx_end = idx_beg + len(model_grp)
                self._source_tf_ids[idx_beg:idx_end] = self.id_maps.bmtk2tf_id_map(src_pop)[model_grp.src_node_ids[()]] # bmtk2tf_id_map[model_grp.src_node_ids[()]]
                self._target_tf_ids[idx_beg:idx_end] = self.id_maps.bmtk2tf_id_map(trg_pop)[model_grp.trg_node_ids[()]] # bmtk2tf_id_map[model_grp.trg_node_ids[()]]
                self._edge_type_ids[idx_beg:idx_end] = model_grp.edge_type_ids
                
                if 'syn_weight' in model_grp.columns:
                    weights[idx_beg:idx_end] = model_grp.get_values('syn_weight', all_rows=True)
                elif 'syn_weight' in et_table.columns:
                    weights[idx_beg:idx_end] = et_table.loc[model_grp.edge_type_ids]['syn_weight']
                else:
                    raise Exception()

                if 'delay' in model_grp.columns:
                    delays[idx_beg:idx_end] = model_grp.get_values('delay', all_rows=True)
                elif 'delay' in et_table.columns:
                    delays[idx_beg:idx_end] = et_table.loc[model_grp.edge_type_ids]['delay']
                else:
                    raise Exception()

                if 'dynamics_params' in model_grp.columns:
                    raise NotImplementedError()
                elif 'dynamics_params' in et_table.columns:
                    syn_ids[idx_beg:idx_end] = syn_dyn_parms_lu[edge_pop.name].loc[model_grp.edge_type_ids]['dyn_params_idx'].values
                else:
                    raise Exception()

        indices = np.column_stack((self._target_tf_ids, self._source_tf_ids))
        sort_order = lex_sort_order_np(indices)
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
            'node_type_ids': np.array(ntids_lu.loc[self._sonata_node_pop.type_ids], dtype=np.int32).flatten(),
            'node_params': {
                'V_th': np.array(self._dynamics_params_lu['V_th'], dtype=np.float32),
                'g': np.array(self._dynamics_params_lu['g'], dtype=np.float32), 
                'E_L': np.array(self._dynamics_params_lu['E_L'], dtype=np.float32), 
                'k': np.array(self._dynamics_params_lu['asc_decay'], dtype=np.float32), 
                'C_m': np.array(self._dynamics_params_lu['C_m'], dtype=np.float32), 
                'V_reset': np.array(self._dynamics_params_lu['V_reset'], dtype=np.float32), 
                't_ref': np.array(self._dynamics_params_lu['t_ref'], dtype=np.float32), 
                'asc_amps': np.array(self._dynamics_params_lu['asc_amps'], dtype=np.float32)
            },
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

        idx_beg, idx_end = 0, 0
        syn_dyn_params, syn_dyn_parms_lu = self._synaptic_dyn_params()
        for edge_pop in self._sonata_edge_pops:
            et_table = edge_pop.types_table.to_dataframe()
            idx_beg = idx_end
            idx_end = idx_beg + len(edge_pop)
            src_pop = edge_pop.source_population 
            trg_pop = edge_pop.target_population
            
            idx_beg = 0
            for model_grp in edge_pop.groups:
                idx_end = idx_beg + len(model_grp)
                self._source_tf_ids[idx_beg:idx_end] = self.id_maps.bmtk2tf_id_map(src_pop)[model_grp.src_node_ids[()]] # bmtk2tf_id_map[model_grp.src_node_ids[()]]
                self._target_tf_ids[idx_beg:idx_end] = self.id_maps.bmtk2tf_id_map(trg_pop)[model_grp.trg_node_ids[()]] # bmtk2tf_id_map[model_grp.trg_node_ids[()]]
                self._edge_type_ids[idx_beg:idx_end] = model_grp.edge_type_ids

                if 'syn_weight' in model_grp.columns:
                    weights[idx_beg:idx_end] = model_grp.get_values('syn_weight', all_rows=True)
                elif 'syn_weight' in et_table.columns:
                    weights[idx_beg:idx_end] = et_table.loc[model_grp.edge_type_ids]['syn_weight']
                else:
                    raise Exception()

                if 'delay' in model_grp.columns:
                    delays[idx_beg:idx_end] = model_grp.get_values('delay', all_rows=True)
                elif 'delay' in et_table.columns:
                    delays[idx_beg:idx_end] = et_table.loc[model_grp.edge_type_ids]['delay']
                else:
                    raise Exception() 

                if 'dynamics_params' in model_grp.columns:
                    raise NotImplementedError()
                elif 'dynamics_params' in et_table.columns:
                    syn_ids[idx_beg:idx_end] = syn_dyn_parms_lu[edge_pop.name].loc[model_grp.edge_type_ids]['dyn_params_idx'].values
                else:
                    raise Exception()

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
