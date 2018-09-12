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
import os
import pandas as pd

class C_Layer (object):
    def __init__(self,node_name,S_Layer_input,bands):
        '''
            :type S_Layer:  S_Layer object
            :param S_Layer:  instance of S_Layer object that serves as input for this C_Layer

            :type bands:  list
            :param bands:  bands[i] = [[list of frequency indices for S_layer over which to pool], grid_size, sample_step]
        '''
        self.node_name = node_name
        self.input = S_Layer_input.input

        self.tf_sess = S_Layer_input.tf_sess

        s_output = S_Layer_input.output

        self.K = S_Layer_input.K

        band_output = {}

        num_bands = len(bands)

        self.band_output = {}

        self.band_shape = {}

        with tf.name_scope(self.node_name):
            for b in range(num_bands):
                bands_to_pool, grid_size, sample_step = bands[b]

                sub_band_shape = []
                for sub_band in bands_to_pool:
                    sub_band_shape += [S_Layer_input.band_shape[sub_band]]

                max_band_shape = sub_band_shape[0]
                for shape in sub_band_shape[1:]:
                    if shape[0] > max_band_shape[0]:  max_band_shape[0] = shape[0]
                    if shape[1] > max_band_shape[1]:  max_band_shape[1] = shape[1]

                # print "max_band_shape = ", max_band_shape
                # for sub_band in bands_to_pool:
                #     print "\tsub_band_shape = ", S_Layer_input.band_shape[sub_band]
                #     print "\tinput band shape = ", s_output[sub_band].get_shape()

                #resize all inputs to highest resolution so that we can maxpool over equivalent scales
                resize_ops = []
                for sub_band in bands_to_pool:
                    op = s_output[sub_band]
    #                resize_ops += [tf.image.resize_images(op,max_band_shape[0],max_band_shape[1],method=ResizeMethod.NEAREST_NEIGHBOR)]
                    resize_ops += [tf.image.resize_nearest_neighbor(op,max_band_shape)]
                    #print "\tresize op shape = ", resize_ops[-1].get_shape()

                #take the maximum for each input channel, element-wise
                max_channel_op = resize_ops[0]
                for op in resize_ops[1:]:
                    max_channel_op = tf.maximum(op,max_channel_op)

                #print "\tmax channel op shape = ", max_channel_op.get_shape()

                # new shape for mode 'SAME'
                # new_band_shape = (max_band_shape[0]/sample_step, max_band_shape[1]/sample_step)
                new_band_shape = np.ceil(np.array(max_band_shape)/float(sample_step)).astype(np.int64)

                # make sure the grid_size and sample_step aren't bigger than the image
                if max_band_shape[0] < grid_size:
                    y_size = max_band_shape[0]
                else:
                    y_size = grid_size

                if max_band_shape[1] < grid_size:
                    x_size = max_band_shape[1]
                else:
                    x_size = grid_size

                if sample_step > max_band_shape[0]:
                    y_step = max_band_shape[0]
                    new_band_shape = (1,new_band_shape[1])
                else:
                    y_step = sample_step
                if sample_step > max_band_shape[1]:
                    x_step = max_band_shape[1]
                    new_band_shape = (new_band_shape[0],1)
                else:
                    x_step = sample_step

                # max pool
                max_pool_op  = tf.nn.max_pool(max_channel_op,[1,y_size,x_size,1],strides=[1,y_step,x_step,1],padding='SAME')

                self.band_shape[b] = new_band_shape
                #print "max_band_shape: ", max_band_shape

                self.band_output[b]=max_pool_op

        self.num_units = 0
        for b in self.band_shape:
            self.num_units += np.prod(self.band_shape[b])*self.K

        self.output = self.band_output

    def __repr__(self):
        return "C_Layer"

    def compute_output(self,X,band):
        return self.tf_sess.run(self.output[band],feed_dict={self.input:X})

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


def test_C1_Layer():

    from S1_Layer import S1_Layer
    import matplotlib.pyplot as plt

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

    # Test c1 on an image
    from isee_engine.mintnet.Image_Library import Image_Library

    image_dir = '/Users/michaelbu/Code/HCOMP/SampleImages'

    im_lib = Image_Library(image_dir)

    image_data = im_lib(1)

    fig, ax = plt.subplots(1)
    ax.imshow(image_data[0,:,:,0],cmap='gray')

    print(image_data.shape)

    fig, ax = plt.subplots(len(bands),len(orientations)*2)
    result = {}
    for b in range(len(bands)):
        result[b] = c1.compute_output(image_data,b)
        print(result[b].shape)
        n, y,x,K = result[b].shape

        for k in range(K):
            #print result[b][i].shape
            # y = i/8
            # x = i%8
            # ax[y,x].imshow(result[b][0,i],interpolation='nearest',cmap='gray')
            # ax[y,x].axis('off')

            ax[b,k].imshow(result[b][0,:,:,k],interpolation='nearest',cmap='gray')
            ax[b,k].axis('off')

    fig.savefig(os.path.join(fig_dir,'c1_layer.tiff'))
    plt.show()

if __name__=='__main__':

    test_C1_Layer()
