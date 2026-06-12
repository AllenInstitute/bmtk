import pandas as pd
import numpy as np
import h5py
from pathlib import Path

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


class ModelWeights:
    def __init__(self, rnn, cell, trainable_only=False, deep_copy=True):
        self.rnn = rnn
        self.cell = cell
        self.trainable_only = trainable_only
        self.deep_copy = deep_copy

        if trainable_only:
            self._networks = {ModelWeights.parse_name(w.name) for w in self.cell.trainable_weights}
        else:
            self._networks = ['<recurrent>'] + self.rnn.inputs_populations
        
        self._network_weights = {}
        for name in self._networks:
            model_vals = cell.recurrent_weight_values if name == '<recurrent>' else self.cell.inputs[name]['input_weight_values']
            model_vals = self._copy_to_host(model_vals) if deep_copy else model_vals
            self._network_weights[name] = model_vals

    @staticmethod
    def _copy_to_host(value):
        if hasattr(value, 'numpy'):
            return value.numpy().copy()
        return np.array(value, copy=True)

    @property
    def networks(self):
        return self._networks

    @staticmethod
    def parse_name(var_name):
        if ':' in var_name:
            var_name = var_name.split(':')[0]
        
        if var_name == 'sparse_recurrent_weights':
            return '<recurrent>'
        else:
            var_name = var_name.replace('_input_weights', '')
            return var_name

    def to_dataframe(self, cache=False):
        ret_df = None
        for netname, netweights in self._network_weights.items():
            netadaptor = self.rnn.get_network(netname)
            conn_table_df = netadaptor.connection_table
            # Recover physical syn_weight from the internally-scaled weights the cell trains on
            # (load divides by voltage_scale[target] * weight_scale / lr_scale; invert it here).
            # Without this the exported SONATA weights are ~voltage_scale too small.
            if netname == '<recurrent>':
                export_factor = self.cell._recurrent_export_factor
            else:
                export_factor = self.cell.inputs[netname]['export_factor']
            netweights_np = netweights.numpy() if hasattr(netweights, 'numpy') else netweights
            conn_table_df['syn_weight'] = np.asarray(netweights_np, dtype=np.float32) * export_factor

            if ret_df is None:
                ret_df = conn_table_df
            else:
                ret_df = pd.concat([ret_df, conn_table_df])

        return ret_df

    def to_sonata(self, output_dir='.', single_file=False, overwrite=False):
        weights_df = self.to_dataframe()
        for (trg_pop, src_pop), conns_df in weights_df.groupby(['target_population', 'source_population']):
            h5_path = Path(output_dir) / f'{src_pop}_{trg_pop}_edges.h5'
            h5_path.parent.mkdir(exist_ok=True, parents=True)

            grp_name = f'/edges/{src_pop}_to_{trg_pop}'
            with h5py.File(h5_path, 'a') as h5_out:
                add_hdf5_magic(h5_out)
                add_hdf5_version(h5_out)
                
                
                if grp_name in h5_out:
                    if overwrite:
                        del h5_out[grp_name]
                    else:
                        raise ValueError(f'SONATA file {h5_path} already exists with group {grp_name}. Please delete or set overwrite=True.')

                edges_grp = h5_out.create_group(grp_name)
                edges_grp.create_dataset('source_node_id', data=conns_df['source_node_id'].values)
                edges_grp['source_node_id'].attrs['node_population'] = src_pop
                edges_grp.create_dataset('target_node_id', data=conns_df['target_node_id'].values)
                edges_grp['target_node_id'].attrs['node_population'] = trg_pop
                edges_grp.create_dataset('edge_type_id', data=conns_df['edge_type_id'].values)
                edges_grp.create_dataset('edge_group_id', data=np.zeros(len(conns_df), dtype=int))
                edges_grp.create_dataset('edge_group_index', data=np.arange(len(conns_df), dtype=int))
                edges_grp.create_dataset('0/syn_weight', data=conns_df['syn_weight'].values)
