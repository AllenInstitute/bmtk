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
import numpy as np
import math
import json
import pandas as pd
import h5py

from neuron import h


def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def edge_converter_csv(output_dir, csv_file):
    """urrently being used by BioNetwork.write_connections(), need to refactor

    :param output_dir:
    :param csv_file:
    :return:
    """
    syns_df = pd.read_csv(csv_file, sep=' ')
    for name, group in syns_df.groupby(['trg_network', 'src_network']):
        trg_net, src_net = name
        group_len = len(group.index)
        with h5py.File(os.path.join(output_dir, '{}_{}_edges.h5'.format(trg_net, src_net)), 'w') as conns_h5:
            conns_h5.create_dataset('edges/target_gid', data=group['trg_gid'])
            conns_h5.create_dataset('edges/source_gid', data=group['src_gid'])
            conns_h5.create_dataset('edges/edge_type_id', data=group['edge_type_id'])
            conns_h5.create_dataset('edges/edge_group', data=group['connection_group'])

            group_counters = {group_id: 0 for group_id in group.connection_group.unique()}
            edge_group_indicies = np.zeros(group_len, dtype=np.uint)
            for i, group_id in enumerate(group['connection_group']):
                edge_group_indicies[i] = group_counters[group_id]
                group_counters[group_id] += 1
            conns_h5.create_dataset('edges/edge_group_indicies', data=edge_group_indicies)

            for group_class, sub_group in group.groupby('connection_group'):
                grp = conns_h5.create_group('edges/{}'.format(group_class))
                if group_class == 0:
                    grp.create_dataset('sec_id', data=sub_group['segment'], dtype='int')
                    grp.create_dataset('sec_x', data=sub_group['section'])
                    grp.create_dataset('syn_weight', data=sub_group['weight'])
                    grp.create_dataset('delay', data=sub_group['delay'])
                elif group_class == 1:
                    grp.create_dataset('syn_weight', data=sub_group['weight'])
                    grp.create_dataset('delay', data=sub_group['delay'])
                else:
                    print('Unknown cell group {}'.format(group_class))
