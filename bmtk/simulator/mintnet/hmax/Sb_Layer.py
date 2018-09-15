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
import tensorflow as tf
from S_Layer import S_Layer
import pandas as pd

class Sb_Layer (object):
    def __init__(self,node_name,C_Layer_input,grid_size,pool_size,K_per_subband,file_name=None):
        '''grid_size is a list, unlike the standard S_Layer, as is file_names'''

        self.node_name = node_name
        self.tf_sess = C_Layer_input.tf_sess

        self.input = C_Layer_input.input

        self.num_sublayers = len(grid_size)
        self.K = K_per_subband*self.num_sublayers  #number of features will be number of sub bands times the K per subband
        self.pool_size = pool_size
        self.grid_size = grid_size

        c_output = C_Layer_input.output

        self.sublayers = {}
        with tf.name_scope(self.node_name):
            for i in range(self.num_sublayers):
                subnode_name = node_name+'_'+str(i)
                self.sublayers[i] = S_Layer(subnode_name,C_Layer_input,grid_size[i],pool_size,K_per_subband,file_name)

            self.band_output = {}
            self.band_shape = C_Layer_input.band_shape

            for band in c_output.keys():

                sub_band_list = []
                for i in range(self.num_sublayers):
                    sub_band_list += [self.sublayers[i].band_output[band]]



                    #gather sub_layer outputs and stack them for each band
                self.band_output[band] = tf.concat(sub_band_list, 3)

        self.output = self.band_output

        self.num_units = 0
        for b in self.band_shape:
            self.num_units += np.prod(self.band_shape[b])*self.K

    def __repr__(self):
        return "Sb_Layer"

    def compute_output(self,X,band):

        return self.tf_sess.run(self.output[band],feed_dict={self.input:X})

    def train(self,image_dir,batch_size=100,image_shape=(256,256)):  #,save_file_prefix='weights'):

        for i in range(self.num_sublayers):
            #save_file = save_file_prefix + '_'+str(i)+'.pkl'

            #try:
            self.sublayers[i].train(image_dir,batch_size,image_shape)   #,save_file)
            #except Exception as e:
            #    print i
            #    raise e

    # def get_compute_ops(self):
    #
    #     node_table = pd.DataFrame(columns=['node','band'])
    #     compute_list = []
    #
    #     for band in self.band_output:
    #         node_table = node_table.append(pd.DataFrame([[self.node_name,band]],columns=['node','band']),ignore_index=True)
    #
    #         compute_list.append(self.output[band])
    #
    #     return node_table, compute_list

    def get_compute_ops(self,unit_table=None):

        compute_list = []

        if unit_table is not None:

            for i, row in unit_table.iterrows():

                if 'y' in unit_table:
                    node, band, y, x = row['node'], int(row['band']), int(row['y']), int(row['x'])
                    compute_list.append(self.output[band][:,y,x,:])

                elif 'band' in unit_table:
                    node, band = row['node'], int(row['band'])
                    compute_list.append(self.output[band])

                else:
                    return self.get_all_compute_ops()

        else:
            return self.get_all_compute_ops()

        return unit_table, compute_list

    def get_all_compute_ops(self):

        compute_list = []
        unit_table = pd.DataFrame(columns=['node','band'])
        for band in self.band_output:
            unit_table = unit_table.append(pd.DataFrame([[self.node_name,band]],columns=['node','band']),ignore_index=True)

            compute_list.append(self.output[band])

        return unit_table, compute_list


def test_S2b_Layer():

    from S1_Layer import S1_Layer
    import matplotlib.pyplot as plt
    from C_Layer import C_Layer

    fig_dir = 'Figures'
    # First we need an S1 Layer
    # these parameters are taken from Serre, et al PNAS for HMAX
    freq_channel_params = [ [7,2.8,3.5],
                            [9,3.6,4.6],
                            [11,4.5,5.6],
                            [13,5.4,6.8],
                            [15,6.3,7.9],
                            [17,7.3,9.1],
                            [19,8.2,10.3],
                            [21,9.2,11.5],
                            [23,10.2,12.7],
                            [25,11.3,14.1],
                            [27,12.3,15.4],
                            [29,13.4,16.8],
                            [31,14.6,18.2],
                            [33,15.8,19.7],
                            [35,17.0,21.2],
                            [37,18.2,22.8],
                            [39,19.5,24.4]]

    orientations = np.arange(4)*np.pi/4

    input_shape = (128,192)
    s1 = S1_Layer(input_shape,freq_channel_params,orientations)

    # Now we need to define a C1 Layer
    bands = [   [[0,1], 8, 3],
                [[2,3], 10, 5],
                [[4,5], 12, 7],
                [[6,7], 14, 8],
                [[8,9], 16, 10],
                [[10,11], 18, 12],
                [[12,13], 20, 13],
                [[14,15,16], 22, 15]]

    c1 = C_Layer(s1,bands)

    print("s1 shape:  ", s1.band_shape)
    print("c1 shape:  ", c1.band_shape)

    grid_size = [6,9,12,15]
    pool_size = 10
    K = 10

    s2b = Sb_Layer(c1,grid_size,pool_size,K)

    print("s2b shape:  ", s2b.band_shape)

    c2b_bands = [    [[0,1,2,3,4,5,6,7],40,40]]

    c2b = C_Layer(s2b,c2b_bands)


    print("c2b shape:  ", c2b.band_shape)
    #print c2b.band_output.keys()
    # Test s2 on an image
    from Image_Library import Image_Library

    image_dir = '/Users/michaelbu/Code/HCOMP/SampleImages'

    im_lib = Image_Library(image_dir,new_size=input_shape)

    image_data = im_lib(1)

    fig, ax = plt.subplots(1)
    ax.imshow(image_data[0,:,:,0],cmap='gray')

    fig,ax = plt.subplots(8,10)

    result = {}
    for b in range(len(bands)):
        result[b] = s2b.compute_output(image_data,b)

        for k in range(K):
            ax[b,k].imshow(result[b][0,:,:,k],interpolation='nearest',cmap='gray')
            ax[b,k].axis('off')

    fig.savefig(os.path.join(fig_dir,'s2b_layer.tiff'))

    fig,ax = plt.subplots(8,10)

    result = {}

    #only one band for c2b
    result[0] = c2b.compute_output(image_data,0)

    for k in range(K):
        ax[b,k].imshow(result[0][0,:,:,k],interpolation='nearest',cmap='gray')
        ax[b,k].axis('off')

    fig.savefig(os.path.join(fig_dir,'c2b_layer.tiff'))


    #plt.show()

    s2b.train(image_dir,batch_size=10,image_shape=input_shape,save_file_prefix='test_S2b_weights')

if __name__=='__main__':

    test_S2b_Layer()
