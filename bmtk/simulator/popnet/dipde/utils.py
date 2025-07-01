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
import math
import warnings
import numpy as np
import pandas as pd
import scipy.interpolate as spinterp
import collections
import h5py
import itertools
import scipy.io as sio
import json
import importlib

"""
Most of these functions are not being used directly by popnet, but may still be used in some other capcity. These have
been marked as depreciated, and should be removed soon.


"""


def get_firing_rate_from_nwb(populations, nwb_file, trial):
    """Calculates firing rates for an external population"""
    h5_file = h5py.File(nwb_file, 'r')
    spike_trains_ds = h5_file['processing'][trial]['spike_train']

    # TODO: look into adding a time window rather than searching for min/max t.
    firing_rates = {}
    for pop in populations:
        spike_counts = []
        spike_min_t = 1.0e30
        spike_max_t = 0.0
        for gid in pop.get_gids():
            spike_train_ds = spike_trains_ds[str(gid)]['data']
            if spike_train_ds is not None and len(spike_train_ds[...]) > 0:
                spike_times = spike_train_ds[...]
                tmp_min = min(spike_times)
                spike_min_t = tmp_min if tmp_min < spike_min_t else spike_min_t
                tmp_max = max(spike_times)
                spike_max_t = tmp_max if tmp_max > spike_max_t else spike_max_t
                spike_counts.append(len(spike_times))

        # TODO make sure t_diffs is not null and spike_counts has some values
        firing_rates[pop.pop_id] = 1.0e03 * np.mean(spike_counts) / (spike_max_t - spike_min_t)
    return firing_rates


def get_firing_rates(populations, spike_trains):
    """Calculates firing rates for an external population"""
    #h5_file = h5py.File(nwb_file, 'r')
    #spike_trains_ds = h5_file['processing'][trial]['spike_train']

    # TODO: look into adding a time window rather than searching for min/max t.
    firing_rates = {}
    for pop in populations:
        spike_counts = []
        spike_min_t = 1.0e30
        spike_max_t = 0.0
        for gid in pop.get_gids():
            spike_times = spike_trains.get_times(gid)
            if spike_times is not None and len(spike_times) > 0:
                tmp_min = min(spike_times)
                spike_min_t = tmp_min if tmp_min < spike_min_t else spike_min_t
                tmp_max = max(spike_times)
                spike_max_t = tmp_max if tmp_max > spike_max_t else spike_max_t
                spike_counts.append(len(spike_times))

        # TODO make sure t_diffs is not null and spike_counts has some values
        firing_rates[pop.pop_id] = 1.0e03 * np.mean(spike_counts) / (spike_max_t - spike_min_t)
    return firing_rates

#############################################
# Depreciated
#############################################
def list_of_dicts_to_dict_of_lists(list_of_dicts, default=None):
    new_dict = {}
    for curr_dict in list_of_dicts:
        print(curr_dict.keys())


#############################################
# Depreciated
#############################################
class KeyDefaultDict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError
        else:
            ret = self[key] = self.default_factory(key)
            return ret


#############################################
# Depreciated
#############################################
def create_firing_rate_server(t, y):

    warnings.warn('Hard coded bug fix for mindscope council 4/27/15')
    t = t/.001/200
    interpolation_callable = spinterp.interp1d(t, y, bounds_error=False, fill_value=0)
    return lambda t: interpolation_callable(t)


#############################################
# Depreciated
#############################################
def create_nwb_server_file_path(nwb_file_name, nwb_path):
    f = h5py.File(nwb_file_name, 'r')
    y = f['%s/data' % nwb_path][:]
    dt = f['%s/data' % nwb_path].dims[0][0].value
    t = np.arange(len(y))*dt
    f.close()
    return create_firing_rate_server(t, y)


#############################################
# Depreciated
#############################################
def get_mesoscale_connectivity_dict():

    # Extract data into a dictionary:
    mesoscale_data_dir = '/data/mat/iSee_temp_shared/packages/mesoscale_connectivity'
    nature_data = {}
    for mat, side in itertools.product(['W', 'PValue'],['ipsi', 'contra']):
        data, row_labels, col_labels = [sio.loadmat(os.path.join(mesoscale_data_dir, '%s_%s.mat' % (mat, side)))[key]
                                        for key in ['data', 'row_labels', 'col_labels']]
        for _, (row_label, row) in enumerate(zip(row_labels, data)):
            for _, (col_label, val) in enumerate(zip(col_labels, row)):
                nature_data[mat, side, str(row_label.strip()), str(col_label.strip())] = val
    
    return nature_data


#############################################
# Depreciated
#############################################
def reorder_columns_in_frame(frame, var):
    varlist = [w for w in frame.columns if w not in var]
    return frame[var+varlist]


#############################################
# Depreciated
#############################################
def population_to_dict_for_dataframe(p):
    
    black_list = ['firing_rate_record', 
                  'initial_firing_rate', 
                  'metadata', 
                  't_record']
    
    json_list = ['p0', 'tau_m']
    
    return_dict = {}
    p_dict = p.to_dict()

    for key, val in p_dict['metadata'].items():
        return_dict[key] = val
    
    for key, val in p_dict.items():
        if key not in black_list:
            if key in json_list:
                val = json.dumps(val)
            return_dict[key] = val
            
    return return_dict


#############################################
# Depreciated
#############################################
def network_dict_to_target_adjacency_dict(network_dict):
    print(network_dict)


#############################################
# Depreciated
#############################################
def population_list_to_dataframe(population_list):
    df = pd.DataFrame({'_tmp': [None]})
    for p in population_list:
        model_dict = {'_tmp': [None]}
        for key, val in population_to_dict_for_dataframe(p).items():
            model_dict.setdefault(key, []).append(val)
        df_tmp = pd.DataFrame(model_dict)

        df = pd.merge(df, df_tmp, how='outer')
    df.drop('_tmp', inplace=True, axis=1)
    return df


#############################################
# Depreciated
#############################################
def df_to_csv(df, save_file_name, index=False, sep=' ', na_rep='None'):
    df.to_csv(save_file_name, index=index, sep=sep, na_rep=na_rep)


#############################################
# Depreciated
#############################################
def population_list_to_csv(population_list, save_file_name):
    df = population_list_to_dataframe(population_list)
    df_to_csv(df, save_file_name)


#############################################
# Depreciated
#############################################
def create_instance(data_dict):
    '''Helper function to create an object from a dictionary containing:

    "module": The name of the module containing the class
    "class": The name of the class to be used to create the object
    '''

    curr_module, curr_class = data_dict.pop('module'), data_dict.pop('class')
    curr_instance = getattr(importlib.import_module(curr_module), curr_class)(**data_dict)

    return curr_instance


#############################################
# Depreciated
#############################################
def assert_model_known(model, model_dict):
    """Test if a model in in the model_dict; if not, raise UnknownModelError"""

    try:
        assert model in model_dict
    except:
        raise Exception('model {} does not exist.'.format(model))


#############################################
# Depreciated
#############################################
def create_population_list(node_table, model_table):
    """Create a population list from the node and model pandas tables"""

    model_dict = {}
    for row in model_table.iterrows():
        model = row[1].to_dict()
        model_dict[model.pop('model')] = model

    population_list = []
    for row in node_table.iterrows():
        node = row[1].to_dict()
        model = node.pop('model')

        # Check if model type in model dict:
        assert_model_known(model, model_dict)

        # Clean up:
        curr_model = {}
        for key, val in model_dict[model].items():
            if not (isinstance(val, float) and math.isnan(val)):
                curr_model[key] = val
        curr_model.setdefault('metadata', {})['model'] = model

        curr_module, curr_class = curr_model['module'], curr_model['class']
        curr_instance = getattr(importlib.import_module(curr_module), curr_class)(**curr_model)
        population_list.append(curr_instance)

    return population_list
