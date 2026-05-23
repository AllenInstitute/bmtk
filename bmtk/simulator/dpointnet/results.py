import numpy as np
import pandas as pd
import h5py
from pathlib import Path


from .id_maps import TFIDMap
from .io_tools import io
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


class _SpikesResults:
    def __init__(self, parent, spikes_table):
        self._spikes_tables = spikes_table
        self.parent = parent

    def to_dataframe(self):
        batch_num, spike_steps, tf_ids = np.nonzero(self._spikes_tables)
        timestamps = spike_steps*self.parent.dt
        
        tf2bmtk_id_map = TFIDMap().tf2bmtk_id_map()
        return pd.DataFrame({
            'batch_num': batch_num,
            'node_ids': tf2bmtk_id_map.loc[tf_ids, 'node_id'].values,
            'timestamps': timestamps,
            'population': tf2bmtk_id_map.loc[tf_ids, 'population'].values
        })
        
    def to_spikes_table(self):
        return self._spikes_tables

    def to_pickle(self, file_path, split_batches=False, ):
        raise NotImplementedError()
    
    def to_npz(self, file_path):
        raise NotImplementedError()

    def to_csv(self, file_path, split_batches=False, overwrite=True):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(file_path).exists() and not overwrite:
            io.log_debug(f'CSV file {file_path} already exists, skip saving. Please delete or set overwrite=True.')
            return

        spikes_df = self.to_dataframe()
        n_batches = len(spikes_df['batch_num'].unique())
        if n_batches == 1:
            if 'batch_num' in spikes_df.columns:
                spikes_df = spikes_df.drop(columns=['batch_num'])
            spikes_df.to_csv(file_path, sep=' ', index=False)
        elif split_batches:
            ext = ''.join(Path(file_path).suffixes)
            for batch_num, batch_spikes_df in spikes_df.groupby('batch_num'):
                bpath = file_path.replace(ext, f'.batch_{batch_num}{ext}')
                bspikes_df = batch_spikes_df.drop(columns='batch_num')
                bspikes_df.to_csv(bpath, sep=' ', index=False)

        else:
            spikes_df.to_csv(file_path, sep=' ', index=False)

    def __to_sonata_helper(self, file_path, spikes_df, overwrite=True):
        with h5py.File(file_path, 'a') as h5:
            add_hdf5_magic(h5)
            add_hdf5_version(h5)
            for pop_name, pop_spikes in spikes_df.groupby('population'):
                grp_name = f'/spikes/{pop_name}'
                if grp_name in h5:
                    if overwrite:
                        del h5[grp_name]
                    else:
                        io.log_warning(f'Group {grp_name} already exists {file_path} and overwrite=False. Not writing spikes!')
                        continue

                pop_grp = h5.create_group(f'/spikes/{pop_name}')
                pop_grp.create_dataset('node_ids', data=pop_spikes['node_ids'])
                pop_grp.create_dataset('timestamps', data=pop_spikes['timestamps'])
                pop_grp['timestamps'].attrs['units'] = 'ms'
                if 'batch_num' in spikes_df.columns:
                    pop_grp.create_dataset('batch_num', data=pop_spikes['batch_num'])

    def to_sonata(self, file_path, split_batches=False, sort_by=None, overwrite=True):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        spikes_df = self.to_dataframe()
        if self.parent.batch_size == 1:
            if 'batch_num' in spikes_df.columns:
                spikes_df = spikes_df.drop(columns=['batch_num'])
            self.__to_sonata_helper(file_path, spikes_df, overwrite=overwrite)

        elif split_batches:
            ext = ''.join(Path(file_path).suffixes)
            for batch_num, batch_spikes_df in spikes_df.groupby('batch_num'):
                bpath = file_path.replace(ext, f'.batch_{batch_num}{ext}')
                bspikes_df = batch_spikes_df.drop(columns='batch_num')
                self.__to_sonata_helper(bpath, bspikes_df, overwrite=overwrite)

        else:
            self.__to_sonata_helper(file_path, spikes_df, overwrite=overwrite)


class _ModelState:
    def __init__(self, parent, model_state, **kwargs):
        pass

    def to_sonata(self, file_path, overwrite=True):
        pass

    def to_pickle(self, file_path, overwrite=True):
        raise NotImplementedError()


class _VoltageResults:
    def __init__(self, parent, voltages_table):
        self.parent = parent
        self.voltages_table = voltages_table

    def __to_sonata_helper(self, file_path, batch_num=None, overwrite=True):
        tf2bmtk_id_map = TFIDMap().tf2bmtk_id_map()
        with h5py.File(file_path, 'a') as h5:
            for pop_name, pop_df in tf2bmtk_id_map.groupby('population'):
                grp_name = f'/report/{pop_name}'
                if grp_name in h5:
                    if overwrite:
                        del h5[grp_name]
                    else:
                        io.log_warning(f'Group {grp_name} already exists {file_path} and overwrite=False. Not writing spikes!')
                        continue

                pop_grp = h5.create_group(grp_name)
                node_idxs = pop_df.index.values
                
                if self.parent.batch_size == 1:
                    batch_data_paths = [(1, 'data')]
                elif batch_num is not None:
                    batch_data_paths = [(batch_num, 'data')]
                else:
                    batch_data_paths = [(batch_num, f'data/batch_{batch_num}') for batch_num in range(self.parent.batch_size)]
                
                for b_num, b_path in batch_data_paths:
                    # data = self.voltages_table[batch_num, :, node_idxs]
                    data = self.voltages_table.numpy()[b_num, :, node_idxs]
                    pop_grp.create_dataset(b_path, data=data.T)

                mapping_grp = pop_grp.create_group('mapping')
                mapping_grp.create_dataset('node_id', data=pop_df['node_id'])
                mapping_grp.create_dataset('time', data=[0.0, self.parent.seq_len*self.parent.dt, self.parent.dt])
                mapping_grp.create_dataset('index_pointer', data=np.arange(len(node_idxs), dtype='int'))
                mapping_grp.create_dataset('element_ids', data=np.zeros(len(node_idxs), dtype='int')) 
                mapping_grp.create_dataset('element_pos', data=np.zeros(len(node_idxs), dtype='int')) 
                
    def to_sonata(self, file_path, split_batches=False, overwrite=True):
        if split_batches:
            ext = ''.join(Path(file_path).suffixes)
            for batch_num in range(self.parent.batch_size):
                batched_file_path = file_path.replace(ext, f'.batch_{batch_num}{ext}')
                self.__to_sonata_helper(batched_file_path, batch_num=batch_num, overwrite=overwrite)

        else:
             self.__to_sonata_helper(file_path, overwrite=overwrite)

    def to_pickle(self, file_path, split_batches=False, overwrite=True):
        raise NotImplementedError()


class RNNExtractorResults:
    def __init__(self, seq_len, dt, batch_size, extractor_results):
        self.seq_len = seq_len
        self.dt = dt
        self.batch_size = batch_size
        # bmtk_ids = TFIDMap().recurrent_bmtk_ids()

        self._extractor_results = extractor_results
        self._spikes = None
        self._voltages = None
        self._model_state = None

    @property
    def spikes(self):
        if self._spikes is None:
            self._spikes = _SpikesResults(self, self._extractor_results[0][0])
        return self._spikes
    
    @property
    def voltages(self):
        if self._voltages is None:
            self._voltages = _VoltageResults(self, self._extractor_results[0][1])
        return self._voltages

    @property
    def model_state(self):
        if self._model_state is None:
            self._model_state = _ModelState(self, self._extractor_results[1])

    def save_results(self, **save_opts):
        output_dir = save_opts.get('output_dir', '.')
        split_batches = save_opts.get('split_batches', False)
        overwrite_results = save_opts.get('overwrite_results', True)

        spikes_output_sonata = save_opts.get('spikes_file', None)
        if spikes_output_sonata:
            file_path = RNNExtractorResults.get_actual_path(output_dir, spikes_output_sonata)
            self.spikes.to_sonata(file_path, split_batches=split_batches, overwrite=overwrite_results)

        spikes_output_csv = save_opts.get('spikes_file_csv', None)
        if spikes_output_csv:
            file_path = RNNExtractorResults.get_actual_path(output_dir, spikes_output_csv)
            self.spikes.to_csv(file_path, split_batches=split_batches, overwrite=overwrite_results)

        spikes_output_pickle = save_opts.get('spikes_file_pkl', None)
        if spikes_output_pickle:
            file_path = RNNExtractorResults.get_actual_path(output_dir, spikes_output_pickle)
            self.spikes.to_pickle(file_path, split_batches=split_batches, overwrite=overwrite_results)

        spikes_output_npz = save_opts.get('spikes_file_npz', None)
        if spikes_output_npz:
            file_path = RNNExtractorResults.get_actual_path(output_dir, spikes_output_npz)
            self.spikes.to_npz(file_path, split_batches=split_batches, overwrite=overwrite_results)

        voltages_output_sonata = save_opts.get('voltages_file', None)
        if voltages_output_sonata:
            file_path = RNNExtractorResults.get_actual_path(output_dir, voltages_output_sonata)
            self.voltages.to_sonata(file_path, split_batches=split_batches, overwrite=overwrite_results)

        voltages_output_pickle = save_opts.get('voltages_file_pkl', None)
        if voltages_output_pickle:
            file_path = RNNExtractorResults.get_actual_path(output_dir, voltages_output_sonata)
            self.voltages.to_pickle(file_path, split_batches=split_batches, overwrite=overwrite_results)

        model_states_sonata = save_opts.get('model_states', None)
        if model_states_sonata:
            file_path = RNNExtractorResults.get_actual_path(output_dir, model_states_sonata)
            self.voltages.to_pickle(file_path, split_batches=split_batches, overwrite=overwrite_results)

        model_states_pickle = save_opts.get('model_states_pkl', None)
        if model_states_pickle:
            file_path = RNNExtractorResults.get_actual_path(output_dir, model_states_pickle)
            self.voltages.to_pickle(file_path, split_batches=split_batches, overwrite=overwrite_results)

    @staticmethod
    def get_actual_path(output_dir, file_path):
        file_path = Path(file_path)
        if file_path.is_absolute():
            return file_path
        elif file_path.is_relative_to(output_dir):
            return file_path
        else:
            return Path(output_dir) / file_path
