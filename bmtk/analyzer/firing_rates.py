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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

from bmtk.utils.sonata.config import SonataConfig


def convert_rates(rates_file):
    rates_df = pd.read_csv(rates_file, sep=' ', names=['gid', 'time', 'rate'])
    rates_sorted_df = rates_df.sort_values(['gid', 'time'])
    rates_dict = {}
    for gid, rates in rates_sorted_df.groupby('gid'):
        start = rates['time'].iloc[0]
        end = rates['time'].iloc[-1]
        dt = float(end - start)/len(rates)
        rates_dict[gid] = {'start': start, 'end': end, 'dt': dt, 'rates': np.array(rates['rate'])}

    return rates_dict


def firing_rates_equal(rates_file1, rates_file2, err=0.0001):
    trial_1 = convert_rates(rates_file1)
    trial_2 = convert_rates(rates_file2)
    if set(trial_1.keys()) != set(trial_2.keys()):
        return False

    for gid, rates_data1 in trial_1.items():
        rates_data2 = trial_2[gid]
        if rates_data1['dt'] != rates_data2['dt'] or rates_data1['start'] != rates_data2['start'] or rates_data1['end'] != rates_data2['end']:
            return False

        for r1, r2 in zip(rates_data1['rates'], rates_data2['rates']):
            if abs(r1 - r2) > err:
                return False

    return True


def plot_rates_popnet(cell_models_file, rates_file, model_keys=None, save_as=None, show_plot=True):
    """Initial method for plotting popnet output

    :param cell_models_file:
    :param rates_file:
    :param model_keys:
    :param save_as:
    :param show_plot:
    :return:
    """

    pops_df = pd.read_csv(cell_models_file, sep=' ')
    lookup_col = model_keys if model_keys is not None else 'node_type_id'
    pop_keys = {str(r['node_type_id']): r[lookup_col] for _, r in pops_df.iterrows()}

    # organize the rates file by population
    # rates = {pop_name: ([], []) for pop_name in pop_keys.keys()}
    rates_df = pd.read_csv(rates_file, sep=' ', names=['id', 'times', 'rates'])
    for grp_key, grp_df in rates_df.groupby('id'):
        grp_label = pop_keys[str(grp_key)]
        plt.plot(grp_df['times'], grp_df['rates'], label=grp_label)

    plt.legend(fontsize='x-small')
    plt.xlabel('time (s)')
    plt.ylabel('firing rates (Hz)')

    if save_as is not None:
        plt.savefig(save_as)

    if show_plot:
        plt.show()


def plot_rates(config_file=None, rates_files=None, population=None, times=None, label_column='node_id', title=None, show=True,
                save_as=None, group_by=None, group_excludes=None,
                nodes_file=None, node_types_file=None, plt_style=None):

    def is_hdf5(file_path):
        try:
            h5py.File(file_path, 'r')
            return True
        except Exception:
            return False

    def is_csv(file_path):
        try:
            pd.read_csv(file_path)
            return True
        except Exception:
            return False


    sonata_config = SonataConfig.from_json(config_file) if config_file else None

    rates_paths = set()
    if isinstance(rates_files, (list, tuple, pd.Series)):
        rates_paths.update([Path(rf).absolute() for rf in rates_files])
    elif rates_files is not None:
        rates_paths.add(Path(rates_files))
    
    if sonata_config is not None:
        for k in ['rates_file', 'rates_file_csv', 'rates_file_h5']:
            if k in sonata_config.output:
                rates_paths.add(Path(sonata_config.get('output', {}).get('output_dir', '.')) / Path(sonata_config.output[k]))
                break
    
    fig, ax = plt.subplots(1, 1)
    for rpath in rates_paths:
        if is_csv(rpath):
            rates_df = pd.read_csv(rpath, sep=' ')
            for (population, node_id), subdf in rates_df.groupby(['population', 'node_id']):
                times = subdf['timestamps'].values
                frs = subdf['firing_rates'].values
                
                if label_column == 'population':
                    label = population
                elif label_column == 'node_id':
                    label = node_id
                elif label_column in subdf.columns:
                    label = subdf[label_column].iloc[0]
                else:
                    label = ''

                ax.plot(times, frs, label=label)
        
        elif is_hdf5(rpath):
            with h5py.File(rpath, 'r') as h5:
                for population, popgrp in h5['/rates'].items():
                    times = popgrp['mapping/time'][()]
                    node_ids = popgrp['mapping/node_ids'][()]

                    if label_column == 'population':
                        labels = [population]*len(node_ids)
                    elif label_column == 'node_id':
                        labels = node_ids
                    elif label_column in popgrp['mapping']:
                        labels = popgrp['mapping'][label_column][()].astype(str)
                    else:
                        labels = ['']*len(node_ids)

                    for idx, node_id in enumerate(node_ids):
                        frs = popgrp['data'][:, idx]
                        ax.plot(times, frs, label=labels[idx])
    

    ax.legend()
    ax.set_ylabel('firing rates (Hz)')
    ax.set_xlabel('times (ms)')
    
    if show:
        plt.show()