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
from bmtk.simulator.mintnet.Image_Library_Supervised import Image_Library_Supervised
import h5py

class Readout_Layer (object):

    def __init__(self,node_name,input_layer,K,lam,alt_image_dir='',file_name=None):

        self.node_name = node_name
        self.K = K
        self.input_layer = input_layer
        self.weight_file = file_name
        self.lam = lam

        self.alt_image_dir = alt_image_dir

        if file_name==None:
            new_weights=True
            self.train_state = False
        else:

            weight_h5 = h5py.File(self.weight_file,'a')
            file_open=True

            if self.node_name in weight_h5.keys():

                new_weights = False
                weight_data = weight_h5[self.node_name]['weights'].value
                self.train_state = weight_h5[self.node_name]['train_state'].value

            else:

                new_weights = True
                self.train_state =False
                weight_h5.create_group(self.node_name)
                weight_h5[self.node_name]['train_state']=self.train_state

        self.input = self.input_layer.input
        #self.tf_sess = self.input_layer.tf_sess
        self.tf_sess = tf.Session()

        self.w_shape = (self.input_layer.K,self.K)

        if new_weights:
            #weights=1.0*np.ones(self.w_shape).astype(np.float32)
            weights=100000*np.random.normal(size=self.w_shape).astype(np.float32)
            if file_name!=None:
                weight_h5[self.node_name].create_dataset('weights',shape=weights.shape,dtype=np.float32,compression='gzip',compression_opts=9)
                weight_h5[self.node_name]['weights'][...]=weights
        else:
            weights=weight_data

        self.weights = tf.Variable(weights.astype(np.float32),trainable=True,name='weights')
        self.weights.initializer.run(session=self.tf_sess)
        self.bias = tf.Variable(np.zeros(self.K,dtype=np.float32),trainable=True,name='bias')
        self.bias.initializer.run(session=self.tf_sess)

        # sigmoid doesn't seem to work well, and is slow
        #self.output = tf.sigmoid(tf.matmul(self.input_layer.output,W)+self.bias)  

        self.input_placeholder = tf.placeholder(tf.float32,shape=(None,self.input_layer.K))
        #self.output = tf.nn.softmax(tf.matmul(self.input_placeholder,self.weights) + self.bias)
        self.linear = tf.matmul(self.input_placeholder,self.weights) #+ self.bias

        self.output = tf.sign(self.linear)
        #self.output = tf.nn.softmax(self.linear)
        #self.output = tf.nn.softmax(tf.matmul(self.input_layer.output,self.weights) + self.bias)

        self.y = tf.placeholder(tf.float32,shape=(None,self.K))


        #self.cost = -tf.reduce_mean(self.y*tf.log(self.output))
        self.cost = tf.reduce_mean((self.y - self.output)**2) + self.lam*(tf.reduce_sum(self.weights))**2

        # not gonna do much with current cost function :)
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cost)

        self.num_units = self.K

        if file_open:
            weight_h5.close()

    def compute_output(self,X):

        #return self.tf_sess.run(self.output,feed_dict={self.input:X})  

        rep = self.input_layer.tf_sess.run(self.input_layer.output,feed_dict={self.input:X})

        return self.tf_sess.run(self.output,feed_dict={self.input_placeholder:rep})

    def predict(self,X):

        y_vals = self.compute_output(X)

        return np.argmax(y_vals,axis=1) 

    def train(self,image_dir,batch_size=10,image_shape=(256,256),max_iter=200):

        print("Training")

        im_lib = Image_Library_Supervised(image_dir,new_size=image_shape)

        # let's use the linear regression version for now
        training_lib_size = 225
        y_vals, image_data = im_lib(training_lib_size,sequential=True)

        y_vals = y_vals.T[0].T
        y_vals = 2*y_vals - 1.0

        print(y_vals)
        # print y_vals
        # print image_data.shape

        # import matplotlib.pyplot as plt
        # plt.imshow(image_data[0,:,:,0])
        # plt.figure()
        # plt.imshow(image_data[1,:,:,0])
        # plt.figure()
        # plt.imshow(image_data[9,:,:,0])

        # plt.show()

        num_batches = int(np.ceil(2*training_lib_size/float(batch_size)))
        rep_list = []
        for i in range(num_batches):
            print(i)
            # if i==num_batches-1:
            #     rep = self.input_layer.tf_sess.run(self.input_layer.output,feed_dict={self.input:image_data[i*batch_size:i*batch_size + training_lib_size%batch_size]})
            # else:    
            rep = self.input_layer.tf_sess.run(self.input_layer.output,feed_dict={self.input:image_data[i*batch_size:(i+1)*batch_size]})
            rep_list += [rep]

        rep = np.vstack(rep_list)


        C = np.dot(rep.T,rep) + self.lam*np.eye(self.input_layer.K)
        W = np.dot(np.linalg.inv(C),np.dot(rep.T,y_vals)).astype(np.float32)

        self.tf_sess.run(self.weights.assign(tf.expand_dims(W,1)))

        train_result = self.tf_sess.run(self.output,feed_dict={self.input_placeholder:rep})

        print(W)
        print(train_result.flatten())
        print(y_vals.flatten())
        #print (train_result.flatten() - y_vals.flatten())
        print("train error = ", np.mean((train_result.flatten() != y_vals.flatten())))

        from scipy.stats import norm
        target_mask = y_vals==1
        dist_mask = np.logical_not(target_mask)
        hit_rate = np.mean(train_result.flatten()[target_mask] == y_vals.flatten()[target_mask])
        false_alarm = np.mean(train_result.flatten()[dist_mask] != y_vals.flatten()[dist_mask])
        dprime = norm.ppf(hit_rate) - norm.ppf(false_alarm)
        print("dprime = ", dprime)

        # Test error
        im_lib = Image_Library_Supervised('/Users/michaelbu/Data/SerreOlivaPoggioPNAS07/Train_Test_Set/Test',new_size=image_shape)

        testing_lib_size = 300
        y_vals_test, image_data_test = im_lib(testing_lib_size,sequential=True)

        y_vals_test = y_vals_test.T[0].T
        y_vals_test = 2*y_vals_test - 1.0

        num_batches = int(np.ceil(2*testing_lib_size/float(batch_size)))
        rep_list = []
        for i in range(num_batches):
            print(i)
            # if i==num_batches-1:
            #     rep = self.input_layer.tf_sess.run(self.input_layer.output,feed_dict={self.input:image_data[i*batch_size:i*batch_size + training_lib_size%batch_size]})
            # else:    
            rep = self.input_layer.tf_sess.run(self.input_layer.output,feed_dict={self.input:image_data_test[i*batch_size:(i+1)*batch_size]})
            rep_list += [rep]

        rep_test = np.vstack(rep_list)

        test_result = self.tf_sess.run(self.output,feed_dict={self.input_placeholder:rep_test})

        #print test_result
        print("test error = ", np.mean((test_result.flatten() != y_vals_test.flatten())))
        target_mask = y_vals_test==1
        dist_mask = np.logical_not(target_mask)
        hit_rate = np.mean(test_result.flatten()[target_mask] == y_vals_test.flatten()[target_mask])
        false_alarm = np.mean(test_result.flatten()[dist_mask] != y_vals_test.flatten()[dist_mask])
        dprime = norm.ppf(hit_rate) - norm.ppf(false_alarm)
        print("dprime = ", dprime)

        print(rep_test.shape)


        # logistic regression unit
        # import time
        # for n in range(max_iter):
        #     start = time.time()
        #     print "\tIteration ", n

        #     y_vals, image_data = im_lib(batch_size,sequential=True)

        #     print "\tComputing representation"
        #     rep = self.input_layer.tf_sess.run(self.input_layer.output,feed_dict={self.input:image_data})

        #     print "\tGradient descent step"
        #     #print "rep shape = ", rep.shape
        #     self.tf_sess.run(self.train_step,feed_dict={self.input_placeholder:rep,self.y:y_vals})

            
        #     #self.tf_sess.run(self.train_step,feed_dict={self.input:image_data,self.y:y_vals})

        #     #print "\t\ttraining batch cost = ", self.tf_sess.run(self.cost,feed_dict={self.input:image_data,self.y:y_vals})

        #     print "\t\tTraining error = ", np.mean(np.abs(np.argmax(y_vals,axis=1) - self.predict(image_data)))
        #     print y_vals
        #     print
        #     print self.predict(image_data)
        #     print "\t\ttraining batch cost = ", self.tf_sess.run(self.cost,feed_dict={self.input_placeholder:rep,self.y:y_vals})
        #     print "\t\ttraining linear model = ", self.tf_sess.run(self.linear,feed_dict={self.input_placeholder:rep,self.y:y_vals})

        #     print "\t\ttotal time = ", time.time() - start

