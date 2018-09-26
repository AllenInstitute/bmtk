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
from scipy.misc import imresize
import os
import pandas as pd

stimulus_folder = os.path.dirname(os.path.abspath(__file__))
bob_stimlus = os.path.join(stimulus_folder,'lsn.npy')

class LocallySparseNoise (object):

    def __init__(self,stim_template=None, stim_table=None):

        if stim_template is None or stim_table is None:
            raise Exception("stim_template or stim_table not provided.  Please provide them or call the class methods .with_new_stimulus or .with_bob_stimulus.")
        else:
            self.stim_template = stim_template
            self.stim_table = stim_table

        T,y,x = stim_template.shape

        self.T = T
        self.y = y
        self.x = x


    def get_image_input(self, new_size=None, add_channels=False):

        if new_size is not None:
            y,x = new_size
            data_new_size = np.empty((self.T,y,x),dtype=np.float32)

            for t in range(self.stim_template.shape[0]):
                data_new_size[t] = imresize(self.stim_template[t].astype(np.float32),new_size,interp='nearest')

        if add_channels:
            return data_new_size[:,:,:,np.newaxis]
        else:
            return data_new_size

    @staticmethod
    def exclude(av,y_x,exclusion=0):
        y, x = y_x
        X,Y = np.meshgrid(np.arange(av.shape[1]), np.arange(av.shape[0]))

        mask = ((X-x)**2 + (Y-y)**2) <= exclusion**2
        av[mask] = False

    @classmethod
    def create_sparse_noise_matrix(cls,Y=16,X=28,exclusion=5,T=9000, buffer_x=6, buffer_y=6):

        Xp = X+2*buffer_x
        Yp = Y+2*buffer_y

        # 127 is mean luminance value
        sn = 127*np.ones([T,Yp,Xp],dtype=np.uint8)

        for t in range(T):
            available = np.ones([Yp,Xp]).astype(np.bool)

            while np.any(available):
                y_available, x_available = np.where(available)

                pairs = zip(y_available,x_available)
                pair_index = np.random.choice(range(len(pairs)))
                y,x = pairs[pair_index]

                p = np.random.random()
                if p < 0.5:
                    sn[t,y,x] = 255
                else:
                    sn[t,y,x] = 0

                cls.exclude(available,(y,x),exclusion=exclusion)

        return sn[:,buffer_y:(Y+buffer_y), buffer_x:(X+buffer_x)]

    def save_to_hdf(self):

        pass

    @staticmethod
    def generate_stim_table(T,start_time=0,trial_length=250):
        '''frame_length is in milliseconds'''

        start_time_array = trial_length*np.arange(T) + start_time
        column_list  = [np.arange(T),start_time_array, start_time_array+trial_length-1]  # -1 is because the tables in BOb use inclusive intervals, so we'll stick to that convention
        cols = np.vstack(column_list).T
        stim_table = pd.DataFrame(cols,columns=['frame','start','end'])

        return stim_table


    @classmethod
    def with_new_stimulus(cls,Y=16,X=28,exclusion=5,T=9000, buffer_x=6, buffer_y=6):

        stim_template = cls.create_sparse_noise_matrix(Y=Y,X=X,exclusion=exclusion,T=T, buffer_x=buffer_x, buffer_y=buffer_y)
        T,y,x = stim_template.shape

        stim_table = cls.generate_stim_table(T)

        new_locally_sparse_noise = cls(stim_template=stim_template, stim_table=stim_table)

        return new_locally_sparse_noise

    @classmethod
    def with_brain_observatory_stimulus(cls):

        stim_template = np.load(bob_stimlus)
        T,y,x = stim_template.shape

        stim_table = cls.generate_stim_table(T)

        new_locally_sparse_noise = cls(stim_template=stim_template, stim_table=stim_table)

        return new_locally_sparse_noise
