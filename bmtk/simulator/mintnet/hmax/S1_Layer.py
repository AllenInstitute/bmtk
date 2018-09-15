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

def gabor(X,Y,lamb,sigma,theta,gamma,phase):

    X_hat = X*np.cos(theta) + Y*np.sin(theta)
    Y_hat = -X*np.sin(theta) + Y*np.cos(theta)

    arg1 = (0.5/sigma**2)*(X_hat**2 + (gamma**2)*Y_hat**2)
    arg2 = (2.0*np.pi/lamb)*X_hat

    return np.exp(-arg1)*np.cos(arg2 + phase)

class S1_Layer (object):
    def __init__(self,node_name,input_shape,freq_channel_params,orientations):  #,num_cores=8):
        '''
            freq_channel_params is a dictionary of features for each S1 channel
                len(freq_channel_params) ==num_bands  freq_channel_params[i] = [pixels,sigma,lambda,stride]
            orientations is a list of angles in radians for each filter
        '''
        #self.tf_sess = tf.Session()

        self.node_name = node_name
#        NUM_CORES = num_cores  # Choose how many cores to use.
#        NUM_CORES = 1
#        self.tf_sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
#                   intra_op_parallelism_threads=NUM_CORES))
        self.tf_sess = tf.Session()
#        print "Warning:  Using hard-coded number of CPU Cores.  This should be changed to auto-configure when TensorFlow has been updated."

        self.input_shape = (None,input_shape[0],input_shape[1],1)
        self.input = tf.placeholder(tf.float32,shape=self.input_shape,name="input")

        #phases = np.array([0, np.pi/2])
        phases = np.array([0.0])  # HMAX uses dense tiling in lieu of phases (make this explicit later)

        num_bands = len(freq_channel_params)
        num_orientations = len(orientations)
        num_phases = len(phases)
        self.K = num_orientations*num_phases  #number of features per band

        #n_output = num_frequency_channels*num_orientations*num_phases

        n_input = 1

        self.band_filters = {}
        self.filter_params = {}
        self.band_output = {}
        self.output = self.band_output
        self.band_shape = {}

        with tf.name_scope(self.node_name):
            for band in range(num_bands):
                pixels, sigma, lamb, stride = freq_channel_params[band]
                self.band_shape[band] = input_shape

                w_shape = np.array([pixels,pixels,n_input,self.K])

                W = np.zeros(w_shape,dtype=np.float32)

                #compute w values from parameters
                gamma = 0.3  # value taken from Serre et al giant HMAX manuscript from 2005
                X,Y = np.meshgrid(np.arange(pixels),np.arange(pixels))
                X = X - pixels/2
                Y = Y - pixels/2

                #self.filter_params[band] = freq_channel_params[band]
                self.filter_params[band] = {'pixels':pixels,'sigma':sigma,'lambda':lamb, 'stride':stride}  #should I add orientations and phases to this?

                for i in range(self.K):

                    ori_i = i%num_orientations
                    phase_i = i/num_orientations

                    theta = orientations[ori_i]
                    phase = phases[phase_i]

                    zero_mask = np.zeros([pixels,pixels],dtype='bool')
                    zero_mask = (X*X + Y*Y > pixels*pixels/4)

                    W[:,:,0,i] = gabor(X,Y,lamb,sigma,theta,gamma,phase)
                    W[:,:,0,i][zero_mask] = 0.0
                    W[:,:,0,i] = W[:,:,0,i]/np.sqrt(np.sum(W[:,:,0,i]**2))

                W = tf.Variable(W,trainable=False,name='W_'+str(band))
                W.initializer.run(session=self.tf_sess)

                self.band_filters[band] = W

                input_norm = tf.reshape(tf.reduce_sum(self.input*self.input,[1,2,3]),[-1,1,1,1])
                normalized_input = tf.div(self.input,tf.sqrt(input_norm))
                self.band_output[band] = tf.nn.conv2d(normalized_input,W,strides=[1,stride,stride,1],padding='SAME')
                self.band_shape[band] = tuple([int(x) for x in self.band_output[band].get_shape()[1:3]])


            self.num_units = 0
            for b in self.band_shape:
                self.num_units += np.prod(self.band_shape[band])*self.K

    def __del__(self):
        self.tf_sess.close()

    def __repr__(self):
        return "S1_Layer"

    def compute_output(self,X,band):

        return self.tf_sess.run(self.output[band],feed_dict={self.input:X})

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

def S1_Layer_test():

    import matplotlib.pyplot as plt

    fig_dir = 'Figures'

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

    #plot filters, make sure they are correct
    fig, ax = plt.subplots(len(orientations),len(freq_channel_params))
    fig2,ax2 = plt.subplots(len(orientations),len(freq_channel_params))
    for i,theta in enumerate(orientations):
        for j,params in enumerate(freq_channel_params):

            #index = j*len(orientations)*2 + i*2

            fil = s1.tf_sess.run(s1.band_filters[j])[:,:,0,i]

            ax[i,j].imshow(fil,interpolation='nearest',cmap='gray')
            ax[i,j].axis('off')

            fil = s1.tf_sess.run(s1.band_filters[j])[:,:,0,i+4]

            ax2[i,j].imshow(fil,interpolation='nearest',cmap='gray')
            ax2[i,j].axis('off')


    from Image_Library import Image_Library

    image_dir = '/Users/michaelbu/Code/HCOMP/SampleImages'

    im_lib = Image_Library(image_dir)

    image_data = im_lib(1)

    fig, ax = plt.subplots(1)
    ax.imshow(image_data[0,:,:,0],cmap='gray')

    import timeit
    #print timeit.timeit('result = s1.compute_output(image_data)','from __main__ import s1',number=10)

    def f():
        for band in range(len(freq_channel_params)):
            s1.compute_output(image_data,band)

    number = 10
    runs = timeit.Timer(f).repeat(repeat=10,number=number)
    print("Average time (s) for output evaluation for ", number, " runs:  ", np.mean(runs)/number, '+/-', np.std(runs)/np.sqrt(number))



    print("Image shape = ", image_data.shape)


    fig_r, ax_r = plt.subplots(len(orientations),len(freq_channel_params))
    fig_r2,ax_r2 = plt.subplots(len(orientations),len(freq_channel_params))

    for j,params in enumerate(freq_channel_params):

        result = s1.compute_output(image_data,j)
        print("result shape = ", result.shape)

        for i,theta in enumerate(orientations):

            #fil = np.zeros([39,39])
            #index = j*len(orientations)*2 + i*2
            #print s1.params[0]

            ax_r[i,j].imshow(result[0,:,:,i],interpolation='nearest',cmap='gray')
            ax_r[i,j].axis('off')

            ax_r2[i,j].imshow(result[0,:,:,i+4],interpolation='nearest',cmap='gray')
            ax_r2[i,j].axis('off')

    fig_r.savefig(os.path.join(fig_dir,'s1_layer_0.tiff'))
    fig_r2.savefig(os.path.join(fig_dir,'s1_layer_1.tiff'))
    plt.show()

    #sess.close()

if __name__=='__main__':

    S1_Layer_test()
