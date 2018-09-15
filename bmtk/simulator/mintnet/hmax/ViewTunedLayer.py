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
#from bmtk.mintnet.Stimulus.NaturalScenes import NaturalScenes
import h5py
import pandas as pd

class ViewTunedLayer (object):
    def __init__(self,node_name,K,alt_image_dir='',*inputs,**keyword_args):

        self.node_name=node_name

        file_name = keyword_args.get('file_name',None)

        self.alt_image_dir = alt_image_dir

        if file_name==None:
            print("No filename given.  Generating new (random) weights for layer ", node_name)
            self.train_state = False
            new_weights=True
        else:

            self.weight_file = file_name
            weight_h5 = h5py.File(self.weight_file,'a')
            file_open=True

            if self.node_name in weight_h5.keys():

                #print "Loading weights for layer ", node_name, " from ", self.weight_file
                new_weights = False
                weight_data = weight_h5[self.node_name]['weights'].value
                self.train_state = weight_h5[self.node_name]['train_state']

            else:

                new_weights=True
                self.train_state=False
                weight_h5.create_group(self.node_name)
                weight_h5[self.node_name]['train_state']=self.train_state

        self.input = inputs[0].input
        self.tf_sess = inputs[0].tf_sess
        #should add a check that all inputs have the same value of inputs[i].input

        self.K = K

        concat_list = []
        total_K = 0

        with tf.name_scope(self.node_name):
            for i, node in enumerate(inputs):

                output_i = node.output

                for b in output_i:
                    shape = node.band_shape[b]

                    num_K = np.prod(shape)*node.K
                    total_K = total_K + num_K
                    #print "shape = ", shape, "  total_K = ", num_K
                    reshape_op = tf.reshape(output_i[b],[-1,num_K])
                    concat_list += [reshape_op]

            self.input_unit_vector = tf.concat(concat_list, 1)  #shape [batch_size, total_K]

            self.w_shape = (total_K,K)
            #weight = np.random.normal(size=self.w_shape).astype(np.float32)
            if new_weights:
                weight = np.zeros(self.w_shape).astype(np.float32)
                weight_h5[self.node_name].create_dataset('weights',shape=weight.shape,dtype=np.float32,compression='gzip',compression_opts=9)
            else:
                weight = weight_data  #ict['ViewTunedWeight']
                assert weight.shape[0]==total_K, "weights from file are not equal to total input size for layer "+self.node_name


            self.weights = tf.Variable(weight,trainable=False,name='weights')
            self.weights.initializer.run(session=self.tf_sess)

            #print self.input_unit_vector.get_shape(), total_K
            #should this be a dictionary for consistency?
            #print "input unit vector shape = ", self.input_unit_vector.get_shape()
            #print "total_K = ", total_K

            input_norm = tf.expand_dims(tf.reduce_sum(self.input_unit_vector*self.input_unit_vector,[1]),1)    #,[-1,total_K])
            normalized_input = tf.div(self.input_unit_vector,tf.sqrt(input_norm))
            self.output = tf.matmul(normalized_input,self.weights)  #/0.01

            # try gaussian tuning curve centered on preferred feature
            # self.output = tf.exp(-0.5*tf.reduce_sum(self.weights - self.input_unit_vector))

        self.num_units = K

        if file_open:
            weight_h5.close()

    def __repr__(self):
        return "ViewTunedLayer"

    def compute_output(self,X):

        return self.tf_sess.run(self.output,feed_dict={self.input:X})

    def train(self,image_dir,batch_size=10,image_shape=(256,256)):  #,save_file=None):

        print("Training")

        im_lib = Image_Library(image_dir,new_size=image_shape)

        #ns_lib = NaturalScenes.with_new_stimulus_from_folder(image_dir, new_size=image_shape, add_channels=True)

        new_weights = np.zeros(self.w_shape,dtype=np.float32)

        num_batches = self.K/batch_size

        for n in range(num_batches):
            #for k in range(self.K):
            print("\t\tbatch: ", n, "  Total features:  ", n*batch_size)
            print("\t\t\tImporting images for batch")
            image_data = im_lib(batch_size,sequential=True)
            print("\t\t\tDone")

            print("\t\t\tComputing responses for batch")
            batch_output = self.tf_sess.run(self.input_unit_vector,feed_dict={self.input:image_data})
            new_weights[:,n*batch_size:(n+1)*batch_size] = batch_output.T

            print("\t\t\tDone")

        if self.K%batch_size!=0:
            last_batch_size = self.K%batch_size
            print("\t\tbatch: ", n+1, "  Total features:  ", (n+1)*batch_size)
            print("\t\t\tImporting images for batch")
            image_data = im_lib(last_batch_size,sequential=True)
            print("\t\t\tDone")

            print("\t\t\tComputing responses for batch")
            batch_output = self.tf_sess.run(self.input_unit_vector,feed_dict={self.input:image_data})
            new_weights[:,-last_batch_size:] = batch_output.T

        new_weights = new_weights/np.sqrt(np.maximum(np.sum(new_weights**2,axis=0),1e-12))

        self.tf_sess.run(self.weights.assign(new_weights))

        print("")
        print("Saving weights to file ", self.weight_file)
        weight_h5 = h5py.File(self.weight_file,'a')
        weight_h5[self.node_name]['weights'][...] = new_weights
        weight_h5[self.node_name]['train_state'][...] = True
        weight_h5.close()

    def get_compute_ops(self,unit_table=None):

        compute_list = []

        if unit_table is not None:
            for i, row in unit_table.iterrows():
                    compute_list = [self.output]

        else:
            unit_table = pd.DataFrame([[self.node_name]], columns=['node'])
            compute_list = [self.output]

        return unit_table, compute_list



def test_ViewTunedLayer():

    from hmouse_test import hmouse

    image_dir = '/Users/michaelbu/Code/H-MOUSE/ILSVRC2015/Data/DET/test'
    image_shape = (256,256)
    weight_file_prefix = 'S2b_weights_500'

    print("Configuring HMAX network")
    hm = hmouse('config/nodes.csv','config/node_types.csv')

    for node in hm.nodes:
        print(node, "  num_units = ", hm.nodes[node].num_units)

    s4 = ViewTunedLayer(10,hm.nodes['c1'],hm.nodes['c2'],hm.nodes['c2b'])  #,hm.nodes['c3'])

    im_lib = Image_Library(image_dir,new_size=image_shape)
    image_data = im_lib(1)

    print(s4.tf_sess.run(tf.shape(s4.input_unit_vector),feed_dict={s4.input:image_data}))
    print(s4.tf_sess.run(tf.shape(s4.weights)))

    print(s4.compute_output(image_data).shape)

    #s4.train(image_dir,batch_size=10,image_shape=image_shape,save_file='s4_test_weights.pkl')




if __name__=='__main__':

    test_ViewTunedLayer()
