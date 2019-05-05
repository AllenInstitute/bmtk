import os
import numpy as np
import pandas as pd
import h5py

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


def create_single_pop_h5():
    h5_file_old = h5py.File('spike_files/spikes.old.h5', 'r')
    node_ids = h5_file_old['/spikes/gids']
    timestamps = h5_file_old['/spikes/timestamps']

    with h5py.File('spike_files/spikes.one_pop.h5', 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        core_grp = h5.create_group('/spikes/v1')
        core_grp.attrs['sorting'] = 'by_time'
        ts_ds = core_grp.create_dataset('timestamps', data=timestamps, dtype=np.float64)
        ts_ds.attrs['units'] = 'milliseconds'
        nids_ds = core_grp.create_dataset('node_ids', data=node_ids, dtype=np.uint64)


def create_multipop_csv(dir_path='/local1/workspace/bmtk/docs/examples/NWB_files'):
    lgn_h5 = h5py.File(os.path.join(dir_path, 'lgn_spikes.nwb'), 'r')
    tw_h5 = h5py.File(os.path.join(dir_path, 'tw_spikes.nwb'), 'r')

    full_df = pd.DataFrame({
        'timestamps': pd.Series(dtype=np.float64),
        'population': pd.Series(dtype=np.string_),
        'node_ids': pd.Series(dtype=np.uint64)
    })

    for pop_name, pop_h5, n_nodes in [('lgn', lgn_h5, 4000), ('tw', tw_h5, 2000)]:
        spike_train_grp = pop_h5['/processing/trial_0/spike_train']
        for node_id in range(n_nodes):
            tmp_df = pd.DataFrame({
                'timestamps': spike_train_grp[str(node_id)]['data'][()],
                'population': pop_name,
                'node_ids': np.uint64(node_id)
            })

            full_df = full_df.append(tmp_df)

    full_df = full_df[['timestamps', 'population', 'node_ids']]
    full_df.to_csv('spike_files/spikes.multipop.csv', sep=' ', index=False)


def create_multipop_h5():
    spikes_df = pd.read_csv('spike_files/spikes.multipop.csv', sep=' ')
    lgn_spikes_df = spikes_df[spikes_df['population'] == 'lgn']
    tw_spikes_df = spikes_df[spikes_df['population'] == 'tw']
    with h5py.File('spike_files/spikes.multipop.h5', 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)

        lgn_grp = h5.create_group('/spikes/lgn')
        lgn_grp.attrs['sorting'] = 'by_id'
        ts_ds = lgn_grp.create_dataset('timestamps', data=lgn_spikes_df['timestamps'], dtype=np.float64)
        ts_ds.attrs['units'] = 'milliseconds'
        lgn_grp.create_dataset('node_ids', data=lgn_spikes_df['node_ids'], dtype=np.uint64)


        tw_grp = h5.create_group('/spikes/tw')
        tw_grp.attrs['sorting'] = 'by_id'
        ts_ds = tw_grp.create_dataset('timestamps', data=tw_spikes_df['timestamps'], dtype=np.float64)
        ts_ds.attrs['units'] = 'milliseconds'
        tw_grp.create_dataset('node_ids', data=tw_spikes_df['node_ids'], dtype=np.uint64)


def create_nwb():
    spikes_df = pd.read_csv('spike_files/spikes.one_pop.csv', sep=' ')
    with h5py.File('spike_files/spikes.onepop.v1.0.nwb', 'w') as h5:
        spikes_grp = h5.create_group('/processing/trial_0/spike_train')
        for node_id in range(14):
            timestamps = spikes_df[spikes_df['node_ids'] == node_id]['timestamps'].values
            data_ds = spikes_grp.create_dataset('{}/data'.format(node_id), data=timestamps, dtype=np.float64)
            data_ds.attrs['dimension'] = 'time'
            data_ds.attrs['unit'] = 'millisecond'



if __name__ == '__main__':
    # create_multipop_csv()
    # create_multipop_h5()
    create_nwb()


