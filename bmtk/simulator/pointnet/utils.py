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
import h5py
from collections import defaultdict
import pandas as pd
import numpy as np
import six
"""
Most of these functions were collected from previous version of pointnet and are no longer tested and tested. However
some functions may still be used by some people internally at AI for running their own simulations. I have marked all
such functions as UNUSED.

I will leave them alone for now but in the future they should be purged or updated.
"""


def read_LGN_activity(trial_num, file_name):
    # UNUSED.
    spike_train_dict = {}
    f5 = h5py.File(file_name, 'r')
    trial_group = f5['processing/trial_{}/spike_train'.format(trial_num)]
    for cid in trial_group.keys():
        spike_train_dict[int(cid)] = trial_group[cid]['data'][...]

    return spike_train_dict


def read_conns(file_name):
    # UNUSED.
    fc = h5py.File(file_name)
    indptr = fc['indptr']
    cell_size = len(indptr) - 1
    print(cell_size)
    conns = {}
    source = fc['src_gids']
    for xin in six.moves.range(cell_size):
        conns[str(xin)] = list(source[indptr[xin]:indptr[xin+1]])

    return conns


def gen_recurrent_csv(num, offset, csv_file):
    # UNUSED.
    conn_data = np.loadtxt(csv_file)
    target_ids = conn_data[:, 0]
    source_ids = conn_data[:, 1]
    weight_scale = conn_data[:, 2]

    pre = []
    cell_num = num
    params = []
    for xin in six.moves.range(cell_num):
        pre.append(xin+offset)
        ind = np.where(source_ids == xin)

        temp_param = {}
        targets = target_ids[ind] + offset
        weights = weight_scale[ind]
        delays = np.ones(len(ind[0]))*1.5
        targets.astype(float)
        weights.astype(float)
        temp_param['target'] = targets
        temp_param['weight'] = weights*1
        temp_param['delay'] = delays
        params.append(temp_param)

    return pre, params


def gen_recurrent_h5(num, offset, h5_file):
    # UNUSED.
    fc = h5py.File(h5_file)
    indptr = fc['indptr']
    cell_size = len(indptr) - 1
    src_gids = fc['src_gids']
    nsyns = fc['nsyns']
    source_ids = []
    weight_scale = []
    target_ids = []
    delay_v = 1.5  # arbitrary value

    for xin in six.moves.range(cell_size):
        target_ids.append(xin)
        source_ids.append(list(src_gids[indptr[xin]:indptr[xin+1]]))
        weight_scale.append(list(nsyns[indptr[xin]:indptr[xin+1]]))
    targets = defaultdict(list)
    weights = defaultdict(list)
    delays = defaultdict(list)

    for xi, xin in enumerate(target_ids):
        for yi, yin in enumerate(source_ids[xi]):
            targets[yin].append(xin)
            weights[yin].append(weight_scale[xi][yi])
            delays[yin].append(delay_v)

    presynaptic = []
    params = []
    for xin in targets:
        presynaptic.append(xin+offset)
        temp_param = {}
        temp_array = np.array(targets[xin])*1.0 + offset
        temp_array.astype(float)
        temp_param['target'] = temp_array
        temp_array = np.array(weights[xin])
        temp_array.astype(float)
        temp_param['weight'] = temp_array
        temp_array = np.array(delays[xin])
        temp_array.astype(float)
        temp_param['delay'] = temp_array
        params.append(temp_param)

    return presynaptic, params


def load_params(node_name, model_name):
    """
    load information regarding nodes and cell_models from csv files

    Parameters
    ----------
    node_name: json file name for node information
    model_name: json file name for neuron model information

    Returns
    -------
    node_info: 2d array of node info read out from the json file
    mode_info: 2d array of model info read out from the json file
    dict_coordinates: dictionary of coordinates. keyword is the node_id and entries are the x,y and z coordinates.
    """
    # UNUSED.
    node = pd.read_csv(node_name, sep=' ', quotechar='"', quoting=0)
    model = pd.read_csv(model_name, sep=' ', quotechar='"', quoting=0)
    node_info = node.values
    model_info = model.values
    # In NEST, cells do not have intrinsic coordinates. So we have to make some virutial links between cells and
    # coordinates
    dict_coordinates = defaultdict(list)

    for xin in six.moves.range(len(node_info)):
        dict_coordinates[str(node_info[xin, 0])] = [node_info[xin, 2], node_info[xin, 3], node_info[xin, 4]]
    return node_info, model_info, dict_coordinates


def load_conns(cnn_fn):
    """
    load information regarding connectivity from csv files

    Parameters
    ----------
    cnn_fn: json file name for connection information

    Returns
    -------
    connection dictionary
    """
    # UNUSED.
    conns = pd.read_csv(cnn_fn, sep=' ', quotechar='"', quoting=0)
    targets = conns.target_label
    sources = conns.source_label
    weights = conns.weight
    delays = conns.delay

    conns_mapping = {}
    for xin in six.moves.range(len(targets)):
        keys = sources[xin] + '-' + targets[xin]
        conns_mapping[keys] = [weights[xin], delays[xin]]

    return conns_mapping
