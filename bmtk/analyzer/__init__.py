# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
from six import string_types
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from bmtk.utils.sonata.config import SonataConfig as cfg


def _get_config(config):
    if isinstance(config, string_types):
        return cfg.from_json(config)
    elif isinstance(config, dict):
        return config
    else:
        raise Exception('Could not convert {} (type "{}") to json.'.format(config, type(config)))

def plot_potential(cell_vars_h5=None, config_file=None, gids=None, show_plot=True, save=False):
    if (cell_vars_h5 or config_file) is None:
        raise Exception('Please specify a cell_vars hdf5 file or a simulation config.')

    if cell_vars_h5 is not None:
        plot_potential_hdf5(cell_vars_h5, gids=gids, show_plot=show_plot,
                            save_as='sim_potential.jpg' if save else None)

    else:
        # load the json file or object
        if isinstance(config_file, string_types):
            config = cfg.from_json(config_file)
        elif isinstance(config_file, dict):
            config = config_file
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(config_file, type(config_file)))

        gid_list = gids or config['node_id_selections']['save_cell_vars']
        for gid in gid_list:
            save_as = '{}_v.jpg'.format(gid) if save else None
            title = 'cell gid {}'.format(gid)
            var_h5 = os.path.join(config['output']['cell_vars_dir'], '{}.h5'.format(gid))
            plot_potential_hdf5(var_h5, title, show_plot, save_as)


def plot_potential_hdf5(cell_vars_h5, gids, title='membrane potential', show_plot=True, save_as=None):
    data_h5 = h5py.File(cell_vars_h5, 'r')
    membrane_trace = data_h5['data']

    time_ds = data_h5['/mapping/time']
    tstart = time_ds[0]
    tstop = time_ds[1]
    x_axis = np.linspace(tstart, tstop, len(membrane_trace), endpoint=True)

    gids_ds = data_h5['/mapping/gids']
    index_ds = data_h5['/mapping/index_pointer']
    index_lookup = {gids_ds[i]: (index_ds[i], index_ds[i+1]) for i in range(len(gids_ds))}
    gids = gids_ds.keys() if gids_ds is None else gids
    for gid in gids:
        var_indx = index_lookup[gid][0]
        plt.plot(x_axis, membrane_trace[:, var_indx], label=gid)

    plt.xlabel('time (ms)')
    plt.ylabel('membrane (mV)')
    plt.title(title)
    plt.legend(markerscale=2, scatterpoints=1)

    if save_as is not None:
        plt.savefig(save_as)

    if show_plot:
        plt.show()


def plot_calcium(cell_vars_h5=None, config_file=None, gids=None, show_plot=True, save=False):
    if (cell_vars_h5 or config_file) is None:
        raise Exception('Please specify a cell_vars hdf5 file or a simulation config.')

    if cell_vars_h5 is not None:
        plot_calcium_hdf5(cell_vars_h5, gids, show_plot=show_plot, save_as='sim_ca.jpg' if save else None)

    else:
        # load the json file or object
        if isinstance(config_file, string_types):
            config = cfg.from_json(config_file)
        elif isinstance(config_file, dict):
            config = config_file
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(config_file, type(config_file)))

        gid_list = gids or config['node_id_selections']['save_cell_vars']
        for gid in gid_list:
            save_as = '{}_v.jpg'.format(gid) if save else None
            title = 'cell gid {}'.format(gid)
            var_h5 = os.path.join(config['output']['cell_vars_dir'], '{}.h5'.format(gid))
            plot_calcium_hdf5(var_h5, title, show_plot, save_as)


def plot_calcium_hdf5(cell_vars_h5, gids, title='Ca2+ influx', show_plot=True, save_as=None):
    data_h5 = h5py.File(cell_vars_h5, 'r')
    cai_trace = data_h5['cai/data']

    time_ds = data_h5['/mapping/time']
    tstart = time_ds[0]
    tstop = time_ds[1]
    x_axis = np.linspace(tstart, tstop, len(cai_trace), endpoint=True)

    gids_ds = data_h5['/mapping/gids']
    index_ds = data_h5['/mapping/index_pointer']
    index_lookup = {gids_ds[i]: (index_ds[i], index_ds[i+1]) for i in range(len(gids_ds))}
    gids = gids_ds.keys() if gids_ds is None else gids
    for gid in gids:
        var_indx = index_lookup[gid][0]
        plt.plot(x_axis, cai_trace[:, var_indx], label=gid)

    #plt.plot(x_axis, cai_trace)
    plt.xlabel('time (ms)')
    plt.ylabel('calcium [Ca2+]')
    plt.title(title)
    plt.legend(markerscale=2, scatterpoints=1)

    if save_as is not None:
        plt.savefig(save_as)

    if show_plot:
        plt.show()


def spikes_table(config_file, spikes_file=None):
    config = _get_config(config_file)
    spikes_file = config['output']['spikes_file']
    spikes_h5 = h5py.File(spikes_file, 'r')
    gids = np.array(spikes_h5['/spikes/gids'], dtype=np.uint)
    times = np.array(spikes_h5['/spikes/timestamps'], dtype=np.float)
    return pd.DataFrame(data={'gid': gids, 'spike time (ms)': times})
    #return pd.read_csv(spikes_ascii, names=['time (ms)', 'cell gid'], sep=' ')


def nodes_table(nodes_file, population):
    # TODO: Integrate into sonata api
    nodes_h5 = h5py.File(nodes_file, 'r')
    nodes_pop = nodes_h5['/nodes'][population]
    root_df = pd.DataFrame(data={'node_id': nodes_pop['node_id'], 'node_type_id': nodes_pop['node_type_id'],
                                 'node_group_id': nodes_pop['node_group_id'],
                                 'node_group_index': nodes_pop['node_group_index']}) #,
                           #index=[nodes_pop['node_group_id'], nodes_pop['node_group_index']])
    root_df = root_df.set_index(['node_group_id', 'node_group_index'])

    node_grps = np.unique(nodes_pop['node_group_id'])
    for grp_id in node_grps:
        sub_group = nodes_pop[str(grp_id)]
        grp_df = pd.DataFrame()
        for hf_key in sub_group:
            hf_obj = sub_group[hf_key]
            if isinstance(hf_obj, h5py.Dataset):
                grp_df[hf_key] = hf_obj

        subgrp_len = len(grp_df)
        if subgrp_len > 0:
            grp_df['node_group_id'] = [grp_id]*subgrp_len
            grp_df['node_group_index'] = range(subgrp_len)
            grp_df = grp_df.set_index(['node_group_id', 'node_group_index'])
            root_df = root_df.join(other=grp_df, how='left')

    return root_df.reset_index(drop=True)


def node_types_table(node_types_file, population):
    return pd.read_csv(node_types_file, sep=' ')