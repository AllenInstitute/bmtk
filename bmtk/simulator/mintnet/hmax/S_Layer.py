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
from bmtk.simulator.mintnet.Image_Library import Image_Library
import os
import h5py
import pandas as pd

class S_Layer (object):
    def __init__(self, node_name, C_Layer_input, grid_size, pool_size, K, file_name=None, randomize=False):
        self.node_name = node_name

        self.input = C_Layer_input.input

        self.tf_sess = C_Layer_input.tf_sess
        #self.input_layer = C_Layer_input
        # c_output should be a dictionary indexed over bands

        c_output = C_Layer_input.output
        self.C_Layer_input = C_Layer_input

        self.K = K
        self.input_K = C_Layer_input.K
        self.grid_size = grid_size
        self.pool_size = pool_size

        self.band_output = {}
        #self.band_filters = {}
        self.band_shape = C_Layer_input.band_shape
        #print self.band_shape

        file_open = False
        if file_name==None:
            self.train_state=False
            new_weights = True
        else:

            self.weight_file = file_name

            weight_h5 = h5py.File(self.weight_file, 'a')
            file_open = True

            if self.node_name in weight_h5.keys():

                new_weights=False
                weight_data = weight_h5[self.node_name]['weights']
                self.train_state = weight_h5[self.node_name]['train_state'].value

            else:

                new_weights=True
                self.train_state = False
                weight_h5.create_group(self.node_name)
                #weight_h5[self.node_name].create_group('weights')
                weight_h5[self.node_name]['train_state']=self.train_state



            # perform checks to make sure weight_file is consistent with the Layer parameters
            # check input bands
            # check grid_size, pool_size, K

        with tf.name_scope(self.node_name):
            #for band in c_output.keys():

            if new_weights:

                # if self.grid_size >= self.band_shape[band][0]:
                #     size_y = self.band_shape[band][0]
                # else:
                #     size_y = grid_size
                # if self.grid_size >= self.band_shape[band][1]:
                #     size_x = self.band_shape[band][1]
                # else:
                #     size_x = grid_size

                w_shape = np.array([self.grid_size,self.grid_size,self.input_K,self.K])

                self.w_shape = w_shape

                w_bound = np.sqrt(np.prod(w_shape[1:]))
                if randomize:
                    W = np.random.uniform(low= -1.0/w_bound, high=1.0/w_bound, size=w_shape).astype(np.float32)
                else:
                    W = np.zeros(w_shape).astype(np.float32)

                if file_name!=None:
                    weight_h5[self.node_name].create_dataset('weights',shape=w_shape,dtype=np.float32)

            else:
                # Need to check that c_output.keys() has the same set of keys that weight_dict is expecting
                W = weight_data.value
                self.w_shape = W.shape




            W = tf.Variable(W,trainable=False,name='W')
            W.initializer.run(session=self.tf_sess)

            #self.band_filters[band]= W
            self.weights = W

            for band in c_output.keys():
                W_slice = W[:self.band_shape[band][0],:self.band_shape[band][1]]

                input_norm = tf.expand_dims(tf.reduce_sum(c_output[band]*c_output[band],[1,2]),1)   #,[-1,1,1,self.input_K])
                input_norm = tf.expand_dims(input_norm,1)
                normalized_input = tf.div(c_output[band],tf.maximum(tf.sqrt(input_norm),1e-12))
                self.band_output[band] = tf.nn.conv2d(normalized_input,W_slice,strides=[1,1,1,1],padding='SAME')

        self.output = self.band_output

        self.num_units = 0
        for b in self.band_shape:
            self.num_units += np.prod(self.band_shape[b])*self.K

        if file_open:
            weight_h5.close()

    def __repr__(self):
        return "S_Layer"

    def compute_output(self,X,band):

        return self.tf_sess.run(self.output[band],feed_dict={self.input:X})

    def find_band_and_coords_for_imprinting_unit(self, imprinting_unit_index):

        cumulative_units = 0
        for band in self.C_Layer_input.output:

            units_in_next_band = int(np.prod(self.C_Layer_input.output[band].get_shape()[1:3]))

            if imprinting_unit_index < cumulative_units + units_in_next_band:
                # found the right band!
                yb, xb = self.C_Layer_input.band_shape[band]

                band_index = imprinting_unit_index - cumulative_units

                y = band_index/xb
                x = band_index%xb
                break
            else:
                cumulative_units += units_in_next_band

        return band, y, x



    def get_total_pixels_in_C_Layer_input(self):

        total = 0

        band_shape = self.C_Layer_input.band_shape
        band_ids = band_shape.keys()
        band_ids.sort()

        for band in band_ids:
            total += np.prod(band_shape[band])

        return total


    def get_patch_bounding_box_and_shift(self,band,y,x):
        y_lower = y - self.grid_size/2
        y_upper = y_lower + self.grid_size

        x_lower = x - self.grid_size/2
        x_upper = x_lower + self.grid_size

        yb, xb = self.C_Layer_input.band_shape[band]

        # compute shifts in lower bound to deal with overlap with the edges
        y_shift_lower = np.max([-y_lower,0])
        x_shift_lower = np.max([-x_lower,0])


        y_lower = np.max([y_lower,0])
        y_upper = np.min([y_upper,yb])

        x_lower = np.max([x_lower,0])
        x_upper = np.min([x_upper,xb])

        y_shift_upper = y_shift_lower + y_upper - y_lower
        x_shift_upper = x_shift_lower + x_upper - x_lower

        return y_lower, y_upper, x_lower, x_upper, y_shift_lower, y_shift_upper, x_shift_lower, x_shift_upper

    def train(self,image_dir,batch_size=100,image_shape=(256,256)):  #,save_file='weights.pkl'):

        print("Training")

        im_lib = Image_Library(image_dir,new_size=image_shape)

        new_weights = np.zeros(self.w_shape).astype(np.float32)


        for k in range(self.K):

            if k%10==0:
                print("Imprinting feature ", k)
            # how to handle the randomly picked neuron; rejection sampling?
            imprinting_unit_index = np.random.randint(self.get_total_pixels_in_C_Layer_input())

            #print "Imprinting unit index ", imprinting_unit_index
            band, y, x = self.find_band_and_coords_for_imprinting_unit(imprinting_unit_index)
            #print "Imprinting unit in band ", band, " at ", (y, x)

            im_data = im_lib(1)

            output = self.C_Layer_input.compute_output(im_data,band)

            # grab weights from chosen unit, save them to new_weights
            y_lower, y_upper, x_lower, x_upper, y_shift_lower, y_shift_upper, x_shift_lower, x_shift_upper = self.get_patch_bounding_box_and_shift(band,y,x)

            w_patch = output[0,y_lower:y_upper,x_lower:x_upper,:].copy()

            #print "(y_lower, y_upper), (x_lower, x_upper) = ", (y_lower, y_upper), (x_lower, x_upper)
            #print "Patch shape = ", w_patch.shape

            patch_size = np.prod(w_patch.shape)
            # print "self.w_shape = ", self.w_shape, " patch_size = ", patch_size, " pool_size = ", self.pool_size
            # print "band, y, x = ", band,y,x

            pool_size = np.min([self.pool_size,patch_size])
            pool_mask_indices = np.random.choice(np.arange(patch_size), size=pool_size, replace=False)
            pool_mask = np.zeros(patch_size,dtype=np.bool)
            pool_mask[pool_mask_indices] = True
            pool_mask.resize(w_patch.shape)
            pool_mask = np.logical_not(pool_mask)  # we want a mask for the indices to zero out

            w_patch[pool_mask] = 0.0

            # will need to enlarge w_patch if the edges got truncated

            new_weights[y_shift_lower:y_shift_upper,x_shift_lower:x_shift_upper,:,k] = w_patch


        # old code starts here
        # num_batches = self.K/batch_size
        # if self.K%batch_size!=0:
        #     num_batches = num_batches+1

        self.tf_sess.run(self.weights.assign(new_weights))
        print()
        print("Saving weights to file in ", self.weight_file)

        weight_h5 = h5py.File(self.weight_file,'a')
        #for band in new_weights:
        weight_h5[self.node_name]['weights'][...] = new_weights
        weight_h5[self.node_name]['train_state'][...]=True

        weight_h5.close()

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


def test_S_Layer_ouput():

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

    grid_size = 3
    pool_size = 10
    K = 10

    s2 = S_Layer('s2',c1,grid_size,pool_size,K,file_name='S_test_file.h5',randomize=False)

    # Test s2 on an image
    image_dir = '/Users/michaelbu/Code/HCOMP/SampleImages'

    im_lib = Image_Library(image_dir,new_size=input_shape)

    image_data = im_lib(1)

    fig, ax = plt.subplots(1)
    ax.imshow(image_data[0,:,:,0],cmap='gray')

    fig,ax = plt.subplots(8,10)

    result = {}
    for b in range(len(bands)):
        result[b] = s2.compute_output(image_data,b)

        for k in range(K):
            ax[b,k].imshow(result[b][0,:,:,k],interpolation='nearest',cmap='gray')
            ax[b,k].axis('off')

    fig.savefig(os.path.join(fig_dir,'s2_layer.tiff'))
    plt.show()

    s2.train(image_dir,batch_size=10,image_shape=input_shape)   #,save_file='test_weights.pkl')




if __name__=='__main__':
    test_S_Layer_ouput()
