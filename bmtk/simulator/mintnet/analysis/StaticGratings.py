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
import numpy as np
import pandas as pd
import h5py
import sys
import os

class StaticGratings (object):

    def __init__(self,data_file_name):

        self.stim_table = pd.read_hdf(data_file_name,'stim_table')
        self.node_table = pd.read_hdf(data_file_name,'node_table')
        self.tunings_file = None

        f = lambda label: self.stim_table.dropna().drop_duplicates([label])[label].sort_values(inplace=False).values

        self.orientations = f('orientation')
        self.spatial_frequencies = f('spatial_frequency')
        self.phases = f('phase')

        self.data_file_name = data_file_name

        data = h5py.File(self.data_file_name,'r')

        self.data_sets = data.keys()
        self.data_sets.remove('stim_table')
        self.data_sets.remove('node_table')
        self.data_sets.remove('stim_template')

        data.close()

    def tuning_matrix(self, response, dtype=np.float32):

        tuning_shape = tuple([len(self.orientations), len(self.spatial_frequencies), len(self.phases)] + list(response.shape[1:]))

        tuning_matrix = np.empty(tuning_shape, dtype=dtype)

        for i,ori in enumerate(self.orientations):
            for j,sf in enumerate(self.spatial_frequencies):
                for k,ph in enumerate(self.phases):

                    index = self.stim_table[(self.stim_table.spatial_frequency==sf) & (self.stim_table.orientation==ori) & (self.stim_table.phase==ph)].index

                    tuning_matrix[i,j,k] = np.mean(response[index],axis=0)

        return tuning_matrix

    def compute_all_tuning(self, dtype=np.float32, force=False):
        self.tunings_file = self.data_file_name[:-3]+'_analysis.ic'
        if os.path.exists(self.tunings_file) and not force:
            print('Using existing tunings file {}.'.format(self.tunings_file))
            return

        output = h5py.File(self.tunings_file,'a')
        data = h5py.File(self.data_file_name,'r')

        for i, data_set in enumerate(self.data_sets):
            sys.stdout.write( '\r{0:.02f}'.format(float(i)*100/len(self.data_sets))+'% done')
            sys.stdout.flush()

            response = data[data_set].value

            tuning = self.tuning_matrix(response, dtype=dtype)

            key = data_set+'/sg/tuning'
            if key in output:
                del output[key]
            output[key] = tuning

        sys.stdout.write( '\r{0:.02f}'.format(float(100))+'% done')
        sys.stdout.flush()

        data.close()

    def get_tunings_file(self):
        if self.tunings_file is None:
            self.compute_all_tuning()

        return h5py.File(self.tunings_file, 'r')