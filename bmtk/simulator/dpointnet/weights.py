import pandas as pd
import numpy as np
import h5py
from pathlib import Path

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version
from ._options import normalize_sonata_compression


class ModelWeights:
    preserves_canonical_edge_order = True

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
        csr_masters = getattr(cell, '_csr_master_weights', False)
        for name in self._networks:
            if deep_copy and csr_masters:
                # Deep copies snapshot canonical rows; live exports keep the CSR-ordered
                # master and to_dataframe() converts it when exporting.
                self._network_weights[name] = self._copy_to_host(cell.canonical_weight_values(name))
                continue
            model_vals = (
                cell.recurrent_weight_values if name == '<recurrent>'
                else self.cell.inputs[name]['input_weight_values']
            )
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
            if not self.deep_copy and getattr(self.cell, '_csr_master_weights', False):
                netweights = self.cell.canonical_weight_values(netname)
            netadaptor = self.rnn.get_network(netname)
            conn_table_df = netadaptor.connection_table
            # Recover physical syn_weight from the internally-scaled weights the cell trains on
            # (load divides by voltage_scale[target] * weight_scale / lr_scale; invert it here).
            # Without this the exported SONATA weights are ~voltage_scale too small.
            if netname == '<recurrent>':
                export_factor = self.cell._recurrent_export_factor
            else:
                export_factor = self.cell.inputs[netname]['export_factor']
            if getattr(self.cell, '_csr_host_lifecycle', False):
                layout = (
                    self.cell.recurrent_master_layout if netname == '<recurrent>'
                    else self.cell.inputs[netname]['master_layout']
                )
                export_factor = layout.host_to_canonical(export_factor)
            netweights_np = netweights.numpy() if hasattr(netweights, 'numpy') else netweights
            conn_table_df['syn_weight'] = np.asarray(netweights_np, dtype=np.float32) * export_factor
            order = netadaptor.canonical_edge_order
            if order is not None:
                conn_table_df = conn_table_df.iloc[order].reset_index(drop=True)

            if ret_df is None:
                ret_df = conn_table_df
            else:
                ret_df = pd.concat([ret_df, conn_table_df])

        return ret_df

    def to_sonata(self, output_dir='.', single_file=False, overwrite=False, compression='gzip'):
        """Export edges with gzip, lzf, no compression, or a gzip level (1-9)."""
        compression = normalize_sonata_compression(compression)
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
                edges_grp.create_dataset('source_node_id', data=conns_df['source_node_id'].values, compression=compression)
                edges_grp['source_node_id'].attrs['node_population'] = src_pop
                edges_grp.create_dataset('target_node_id', data=conns_df['target_node_id'].values, compression=compression)
                edges_grp['target_node_id'].attrs['node_population'] = trg_pop
                edges_grp.create_dataset('edge_type_id', data=conns_df['edge_type_id'].values, compression=compression)
                edges_grp.create_dataset('edge_group_id', data=np.zeros(len(conns_df), dtype=int), compression=compression)
                edges_grp.create_dataset('edge_group_index', data=np.arange(len(conns_df), dtype=int), compression=compression)
                edges_grp.create_dataset('0/syn_weight', data=conns_df['syn_weight'].values, compression=compression)


class PartitionedModelWeights(ModelWeights):
    """Whole-network weights of a neuron-partitioned run, gathered on rank 0.

    Construction is collective: every rank contributes the physical weights of
    its edges (CSR masters converted to canonical order and multiplied by their
    export factor, the same per-edge arithmetic as :class:`ModelWeights`).
    Rank 0 then holds the recurrent weights in global canonical order and each
    input's weights in file row order and exports them like an unpartitioned
    model; other ranks hold nothing.
    """

    def __init__(self, rnn, cell, trainable_only=False, deep_copy=True):
        from .neuron_partition import PARTITION_INPUT

        partition = rnn.neuron_partition
        self.rnn = rnn
        self.cell = cell
        self.trainable_only = trainable_only
        self.deep_copy = True
        names = ['<recurrent>'] + list(self.rnn.inputs_populations)
        if trainable_only:
            names = [
                name for name in names
                if (cell.recurrent_weight_values if name == '<recurrent>'
                    else cell.inputs[name]['input_weight_values']).trainable
            ]
        self._networks = names
        self.is_root = partition.rank == 0
        self._network_weights = {}
        for name in names:
            if name == '<recurrent>':
                values = partition.gather_canonical({
                    master: self._physical(cell, master) for master in ('<recurrent>', PARTITION_INPUT)
                })
            else:
                rows = partition.comm.gather((partition.input_rows[name], self._physical(cell, name)), root=0)
                values = None
                if self.is_root:
                    values = np.empty(sum(len(r) for r, _ in rows), dtype=np.float32)
                    for peer_rows, peer_values in rows:
                        values[peer_rows] = peer_values
            if self.is_root:
                self._network_weights[name] = values

    @staticmethod
    def _physical(cell, name):
        weights = np.asarray(cell.canonical_weight_values(name).numpy(), dtype=np.float32)
        if name == '<recurrent>':
            export_factor = cell._recurrent_export_factor
            layout = getattr(cell, 'recurrent_master_layout', None)
        else:
            export_factor = cell.inputs[name]['export_factor']
            layout = cell.inputs[name].get('master_layout')
        if getattr(cell, '_csr_host_lifecycle', False) and layout is not None:
            export_factor = layout.host_to_canonical(export_factor)
        return weights * export_factor

    def to_dataframe(self, cache=False):
        if not self.is_root:
            raise ValueError("Partitioned model weights are held by rank 0 only.")
        ret_df = None
        for netname, physical in self._network_weights.items():
            netadaptor = self.rnn.get_network(netname)
            netadaptor.prepare_export_tables()
            conn_table_df = netadaptor.connection_table
            conn_table_df['syn_weight'] = physical
            order = netadaptor.canonical_edge_order
            if order is not None:
                conn_table_df = conn_table_df.iloc[order].reset_index(drop=True)
            ret_df = conn_table_df if ret_df is None else pd.concat([ret_df, conn_table_df])
        return ret_df


class RankModelWeights:
    """This rank's part of a neuron-partitioned network's weights (``weights_export="per_rank"``).

    Construction is local: no rank gathers another rank's edges. :meth:`to_sonata`
    writes one SONATA edges file per network on every rank (population
    ``{source}_to_{target}_rank{r}``), holding the edges into the rank's neurons with
    their physical ``syn_weight``, edge type and the source files' ``delay``; recurrent
    edges are in canonical (target, source) order. Together the ranks' files hold the
    whole network, and they can be listed as ``edges`` of a BMTK config with the
    original edge-types tables. Rank 0 also writes ``weights_parts.json``.
    """

    preserves_canonical_edge_order = True

    def __init__(self, rnn, cell, trainable_only=False, deep_copy=True):
        from .neuron_partition import PARTITION_INPUT

        partition = rnn.neuron_partition
        self.rnn, self.cell = rnn, cell
        self.rank, self.size = partition.rank, partition.size
        names = ['<recurrent>'] + list(rnn.inputs_populations)
        if trainable_only:
            names = [
                name for name in names
                if (cell.recurrent_weight_values if name == '<recurrent>'
                    else cell.inputs[name]['input_weight_values']).trainable
            ]
        self._networks = names
        self._parts = {}
        for name in names:
            adaptor = rnn.get_network(name)
            if name == '<recurrent>':
                masters = ('<recurrent>', PARTITION_INPUT)
                ids = np.concatenate([partition.canonical_ids[m] for m in masters])
                order = np.argsort(ids, kind='stable')
                part = {
                    'target': np.concatenate([adaptor._partition_endpoints[m][0] for m in masters])[order],
                    'source': np.concatenate([adaptor._partition_endpoints[m][1] for m in masters])[order],
                    'edge_type_id': np.concatenate([adaptor._partition_edge_types[m] for m in masters])[order],
                    'delay': np.concatenate([adaptor._partition_delays[m] for m in masters])[order],
                    'syn_weight': np.concatenate(
                        [PartitionedModelWeights._physical(cell, m) for m in masters])[order],
                }
                source_pop = target_pop = adaptor.population_name
            else:
                edges = adaptor.read_edges(keep_target=partition.partition.owner == partition.rank)
                if not np.array_equal(edges['rows'], partition.input_rows[name]):
                    raise ValueError(f'Input {name!r} edges changed since the partition was built.')
                part = {'target': edges['target'], 'source': edges['source'],
                        'edge_type_id': edges['edge_type_ids'], 'delay': edges['delays'],
                        'syn_weight': PartitionedModelWeights._physical(cell, name)}
                source_pop, target_pop = adaptor.population_name, rnn.recurrent_network['name']
            self._parts[name] = (source_pop, target_pop, part)

    @property
    def networks(self):
        return self._networks

    @staticmethod
    def file_name(source_pop, target_pop, rank):
        return f'{source_pop}_{target_pop}_edges.rank{rank}.h5'

    def to_pickle(self, path):
        raise NotImplementedError('Per-rank partition weights are exported with to_sonata().')

    def to_sonata(self, output_dir='.', single_file=False, overwrite=False, compression='gzip'):
        """Collective-free: every rank writes its own files; rank 0 also writes the manifest."""
        import json
        from .id_maps import TFIDMap

        compression = normalize_sonata_compression(compression)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        id_map = TFIDMap()
        manifest = []
        for name, (source_pop, target_pop, part) in self._parts.items():
            h5_path = output_dir / self.file_name(source_pop, target_pop, self.rank)
            if h5_path.exists() and not overwrite:
                raise ValueError(f'SONATA file {h5_path} already exists. Please delete or set overwrite=True.')
            target_ids = id_map.tf2bmtk_id_map(populations=[target_pop])['node_id'].to_numpy()
            source_ids = id_map.tf2bmtk_id_map(populations=[source_pop])['node_id'].to_numpy()
            target_offset = id_map._bmtk_populations[target_pop]['tf_ids'][0]
            source_offset = id_map._bmtk_populations[source_pop]['tf_ids'][0]
            n = len(part['target'])
            population = f'{source_pop}_to_{target_pop}_rank{self.rank}'
            with h5py.File(h5_path, 'w') as h5_out:
                add_hdf5_magic(h5_out)
                add_hdf5_version(h5_out)
                group = h5_out.create_group(f'/edges/{population}')
                options = dict(compression=compression)
                group.create_dataset('source_node_id', data=source_ids[np.asarray(part['source']) - source_offset]
                                     .astype(np.uint64), **options)
                group['source_node_id'].attrs['node_population'] = source_pop
                group.create_dataset('target_node_id', data=target_ids[np.asarray(part['target']) - target_offset]
                                     .astype(np.uint64), **options)
                group['target_node_id'].attrs['node_population'] = target_pop
                group.create_dataset('edge_type_id', data=np.asarray(part['edge_type_id'], np.uint32), **options)
                group.create_dataset('edge_group_id', data=np.zeros(n, dtype=np.uint32), **options)
                group.create_dataset('edge_group_index', data=np.arange(n, dtype=np.uint64), **options)
                group.create_dataset('0/syn_weight', data=np.asarray(part['syn_weight'], np.float32), **options)
                group.create_dataset('0/delay', data=np.asarray(part['delay'], np.float32), **options)
            manifest.append({'network': name, 'source_population': source_pop,
                             'target_population': target_pop})
        if self.rank == 0:
            files = [{**item, 'files': [self.file_name(item['source_population'], item['target_population'], r)
                                        for r in range(self.size)]} for item in manifest]
            (output_dir / 'weights_parts.json').write_text(
                json.dumps({'ranks': self.size, 'networks': files}, indent=1) + '\n')
