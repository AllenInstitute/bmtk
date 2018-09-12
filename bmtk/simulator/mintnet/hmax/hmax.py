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
import sys
import json
from S1_Layer import S1_Layer
from C_Layer import C_Layer
from S_Layer import S_Layer
from Sb_Layer import Sb_Layer
from ViewTunedLayer import ViewTunedLayer
from Readout_Layer import Readout_Layer
import tensorflow as tf
import os
import h5py
import pandas as pd

from bmtk.simulator.mintnet.Image_Library import Image_Library
import matplotlib.pyplot as plt

class hmax (object):

    def __init__(self, configuration, name=None):  #,num_cores=8):
        self.name = name

        if os.path.isdir(configuration):
            # If configuration is a directory look for a config-file inside it.
            self.config_file = os.path.join(configuration, 'config_' + configuration + '.json')
            if self.name is None:
                self.name = os.path.basename(configuration)

        elif os.path.isfile(configuration):
            # If configuration is a json file
            if self.name is None:
                raise Exception("A name is required for configuration parameters")
            self.config_file = configuration

        with open(self.config_file,'r') as f:
            self.config_data = json.loads(f.read())

        self.config_dir = os.path.dirname(os.path.abspath(configuration))
        self.train_state_file = self.__get_config_file(self.config_data['train_state_file'])
        self.image_dir = self.__get_config_file(self.config_data['image_dir'])

        # Find, and create if necessary, the output directory
        if 'output_dir' in self.config_data:
            self.output_dir = self.__get_config_file(self.config_data['output_dir'])
        else:
            self.output_dir = os.path.join(self.config_dir, 'output')

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        with open(self.train_state_file, 'r') as f:
            self.train_state = json.loads(f.read())

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # get the nodes
        models_file = self.__get_config_file(self.config_data['network']['node_types'])
        nodes_file = self.__get_config_file(self.config_data['network']['nodes'])
        self.__nodes_table = self.__build_nodes_table(nodes_file, models_file, self.config_data)

        # Read the connections
        self.nodes = {}
        self.train_order = []

        edges_file = self.__get_config_file(self.config_data['network']['edges'])
        for (node_name, input_node, node_dict) in self.__get_edges(edges_file, self.config_data):
            model_class = self.__nodes_table[node_name]['model_id']

            print("Constructing node:  ", node_name)
            if model_class=='S1_Layer':
                node_type = S1_Layer
                freq_channel_params = node_dict['freq_channel_params']
                input_shape = node_dict['input_shape']
                self.input_shape = input_shape
                orientations = node_dict['orientations']

                self.nodes[node_name] = node_type(node_name,input_shape,freq_channel_params,orientations)  #,num_cores=num_cores)
                #writer = tf.train.SummaryWriter('tmp/hmax', self.nodes['s1'].tf_sess.graph_def)
                #merged = tf.merge_all_summaries()

                #writer.add_summary(self.nodes[node_name].tf_sess.run(merged),0)

            elif model_class=='C_Layer':
                node_type = C_Layer
                bands = node_dict['bands']


                self.nodes[node_name] = node_type(node_name,self.nodes[input_node],bands)
                #writer = tf.train.SummaryWriter('tmp/hmax', self.nodes['s1'].tf_sess.graph_def)

            elif model_class=='S_Layer':
                node_type = S_Layer
                K = node_dict['K']
                weight_file = self.__get_config_file(node_dict['weight_file']) if 'weight_file' in node_dict else None
                pool_size = node_dict['pool_size']
                grid_size = node_dict['grid_size']
                self.train_order += [node_name]

                self.nodes[node_name] = node_type(node_name, self.nodes[input_node], grid_size, pool_size,K,
                                                  file_name=weight_file)

            elif model_class=='Sb_Layer':
                node_type = Sb_Layer
                K = node_dict['K']
                weight_file = self.__get_config_file(node_dict['weight_file']) if 'weight_file' in node_dict else None
                pool_size = node_dict['pool_size']
                grid_size = node_dict['grid_size']

                self.train_order += [node_name]

                self.nodes[node_name] = node_type(node_name,self.nodes[input_node],grid_size,pool_size,K,file_name=weight_file)

            elif model_class=='ViewTunedLayer':
                node_type = ViewTunedLayer
                K = node_dict['K']
                input_nodes = node_dict['inputs']
                input_nodes = [self.nodes[node] for node in input_nodes]
                weight_file = self.__get_config_file(node_dict['weight_file']) if 'weight_file' in node_dict else None
                alt_image_dir = node_dict['alt_image_dir']

                self.train_order += [node_name]

                #print "alt_image_dir=",alt_image_dir
                self.nodes[node_name] = node_type(node_name,K,alt_image_dir,*input_nodes,file_name=weight_file)

            elif model_class=='Readout_Layer':
                node_type = Readout_Layer
                K = node_dict['K']
                input_nodes = self.nodes[input_node]
                weight_file = os.path.join(config_dir,node_dict['weight_file'])
                if weight_file=='':  weight_file=None
                alt_image_dir = node_dict['alt_image_dir']
                lam = node_dict['lam']

                self.train_order += [node_name]

                self.nodes[node_name] = node_type(node_name,self.nodes[input_node],K,lam,alt_image_dir,file_name=weight_file)

            else:
                raise Exception("Unknown model class {}".format(model_class))

            # print "Done"
            # print

        #nfhandle.close()



        self.node_names = self.nodes.keys()

        self.input_shape = (self.nodes['s1'].input_shape[1], self.nodes['s1'].input_shape[2])

        print("Done")
        #writer = tf.train.SummaryWriter('tmp/hmax', self.nodes['s1'].tf_sess.graph_def)


    def __build_nodes_table(self, nodes_csv, models_csv, config):
        models_df = pd.read_csv(models_csv, sep=' ')
        nodes_df = pd.read_csv(nodes_csv, sep=' ')
        nodes_df.set_index('id')
        nodes_full = pd.merge(left=nodes_df, right=models_df, on='model_id')
        nodes_table = {r['id']: {'model_id': r['model_id'], 'python_object': r['python_object']}
                       for _, r in nodes_full.iterrows() }

        return nodes_table

    def __get_edges(self, edges_csv, config):
        def parse_query(query_str):
            if query_str == '*' or query_str == 'None':
                return None
            elif query_str.startswith('id=='):
                return query_str[5:-1]
            else:
                raise Exception('Unknown query string {}'.format(query_str))

        # location where config files are located
        params_dir = self.__get_config_file(config.get('node_config_dir', ''))

        edges_df = pd.read_csv(edges_csv, sep=' ')
        edges = []
        for _, row in edges_df.iterrows():
            # find source and target
            source = parse_query(row['source_query'])
            target = parse_query(row['target_query'])

            # load the parameters from the file
            params_file = os.path.join(params_dir, row['params_file'])
            params = json.load(open(params_file, 'r'))

            # Add to list
            edges.append((target, source, params))

        # TODO: check list and reorder to make sure the layers are in a valid order

        # return the edges. Should we use a generator?
        return edges

    def __get_config_file(self, fpath):
        if os.path.isabs(fpath):
            return fpath
        else:
            return os.path.join(self.config_dir, fpath)



    @classmethod
    def load(cls, config_dir, name=None):
        return cls(config_dir, name)

    def train(self):  #,alt_image_dict=None):

        for node in self.train_order:
            if not self.train_state.get(node, False):
                print("Training Node:  ", node)

                if hasattr(self.nodes[node],'alt_image_dir') and self.nodes[node].alt_image_dir!='':
                    print("\tUsing alternate image directory:  ",  self.nodes[node].alt_image_dir)  # alt_image_dict[node]
                    self.nodes[node].train(self.nodes[node].alt_image_dir,batch_size=self.config_data['batch_size'],image_shape=self.input_shape)
                    self.train_state[node]=True
                else:
                    print("\tUsing default image directory:  ", self.image_dir)
                    self.nodes[node].train(self.image_dir,batch_size=self.config_data['batch_size'],image_shape=self.input_shape)
                    self.train_state[node]=True


                # if node not in alt_image_dict:
                #     print "\tUsing default image directory:  ", image_dir
                #     self.nodes[node].train(image_dir,batch_size=self.config_data['batch_size'],image_shape=self.input_shape)
                #     self.train_state[node]=True
                # else:
                #     print "\tUsing alternate image directory:  ", alt_image_dict[node]
                #     self.nodes[node].train(alt_image_dict[node],batch_size=self.config_data['batch_size'],image_shape=self.input_shape)
                #     self.train_state[node]=True

                print("Done")

            with open(self.config_data['train_state_file'], 'w') as f:
                f.write(json.dumps(self.train_state))


    def run_stimulus(self,stimulus, node_table=None, output_file='output'):
        '''stimulus is an instance of one of the mintnet.Stimulus objects, i.e. LocallySparseNoise'''

        if output_file[-3:]!=".ic":
            output_file = output_file+".ic"  # add *.ic suffix if not already there

        stim_template = stimulus.get_image_input(new_size=self.input_shape, add_channels=True)

        print("Creating new output file:  ", output_file, " (and removing any previous one)")
        if os.path.exists(output_file):
            os.remove(output_file)
        output_h5 = h5py.File(output_file,'w')

        T, y, x, K = stim_template.shape
        all_nodes = self.nodes.keys()

        if node_table is None:  # just compute everything and return it all; good luck!

            new_node_table = pd.DataFrame(columns=['node','band'])

            compute_list = []
            for node in all_nodes:

                add_to_node_table, new_compute_list = self.nodes[node].get_compute_ops()
                new_node_table = new_node_table.append(add_to_node_table,ignore_index=True)
                compute_list += new_compute_list
        else:
            compute_list = []

            new_node_table = node_table.sort_values('node')
            new_node_table = new_node_table.reindex(np.arange(len(new_node_table)))

            for node in all_nodes:
                unit_table = new_node_table[node_table['node']==node]
                if (new_node_table['node']==node).any():
                    _, new_compute_list = self.nodes[node].get_compute_ops(unit_table=unit_table)

                    compute_list += new_compute_list


        # create datasets in hdf5 file from node_table, with data indexed by table index
        for i, row in new_node_table.iterrows():

            output_shape = tuple([T] + [ int(x) for x in compute_list[i].get_shape()[1:]])
            output_h5.create_dataset(str(i), output_shape, dtype=np.float32)



        batch_size = self.config_data['batch_size']
        num_batches = T/batch_size
        if T%self.config_data['batch_size']!=0:
            num_batches += 1

        for i in range(num_batches):
            sys.stdout.write( '\r{0:.02f}'.format(float(i)*100/num_batches)+'% done')
            sys.stdout.flush()
            output_list = self.nodes[all_nodes[0]].tf_sess.run(compute_list,feed_dict={self.nodes[all_nodes[0]].input: stim_template[i*batch_size:(i+1)*batch_size]})

            for io, output in enumerate(output_list):
                # dataset_string = node_table['node'].loc[io] + "/" + str(int(node_table['band'].loc[io]))
                # output_h5[dataset_string][i*batch_size:(i+1)*batch_size] = output

                output_h5[str(io)][i*batch_size:(i+1)*batch_size] = output
        sys.stdout.write( '\r{0:.02f}'.format(float(100))+'% done')
        sys.stdout.flush()

        output_h5['stim_template'] = stimulus.stim_template
        output_h5.close()
        new_node_table.to_hdf(output_file,'node_table')
        if hasattr(stimulus,'label_dataframe') and stimulus.label_dataframe is not None:
            stimulus.label_dataframe.to_hdf(output_file,'labels')
        stimulus.stim_table.to_hdf(output_file,'stim_table')


    def get_exemplar_node_table(self):

        node_table = pd.DataFrame(columns=['node','band','y','x'])
        for node in self.nodes:
            node_output = self.nodes[node].output
            if hasattr(self.nodes[node],'band_shape'):
                for band in node_output:
                    y,x = [int(x) for x in node_output[band].get_shape()[1:3]]
                    y /= 2
                    x /= 2
                    new_row = pd.DataFrame([[self.nodes[node].node_name, band, y, x]], columns=['node','band','y','x'])
                    node_table = node_table.append(new_row, ignore_index=True)
            else:
                new_row = pd.DataFrame([[self.nodes[node].node_name]], columns=['node'])
                node_table = node_table.append(new_row, ignore_index=True)

        return node_table


    def generate_output(self):
        try:
            im_lib = Image_Library(self.image_dir,new_size=self.input_shape)
        except OSError as e:
            print('''A repository of images (such as a collection from ImageNet - http://www.image-net.org) is required for input.
                An example would be too large to include in the isee_engine itself.
                Set the path for this image repository in hmax/config_hmax.json''')
            raise e

        image_data = im_lib(1)

        fig, ax = plt.subplots(1)
        ax.imshow(image_data[0,:,:,0],cmap='gray')

        fig.savefig(os.path.join(self.output_dir,'input_image'))
        plt.close(fig)

        nodes = self.nodes

        for node_to_plot in nodes:
            print("Generating output for node ", node_to_plot)
            node_output_dir = os.path.join(self.output_dir,node_to_plot)

            if not os.path.exists(node_output_dir):
                os.makedirs(node_output_dir)

            if type(self.nodes[node_to_plot])==ViewTunedLayer:
                print("ViewTunedLayer")
                self.nodes[node_to_plot].compute_output(image_data)
                continue

            if type(self.nodes[node_to_plot])==Readout_Layer:
                print("Readout_Layer")
                self.nodes[node_to_plot].compute_output(image_data)
                continue

            num_bands = len(nodes[node_to_plot].output)

            if type(self.nodes[node_to_plot])==S1_Layer or node_to_plot=='c1':
                #print "Yes, this is an S1_Layer"
                num_filters_to_plot = 4
                fig, ax = plt.subplots(num_filters_to_plot,num_bands,figsize=(20,8))
                #fig2,ax2 = plt.subplots(num_filters_to_plot,num_bands,figsize=(20,8))
            else:
                num_filters_to_plot = 8
                fig, ax = plt.subplots(num_filters_to_plot,num_bands,figsize=(20,8))

            for band in range(num_bands):
                result = nodes[node_to_plot].compute_output(image_data,band)
                #print result[band].shape
                n, y,x,K = result.shape

                for k in range(num_filters_to_plot):

                    if num_bands!=1:
                        ax[k,band].imshow(result[0,:,:,k],interpolation='nearest',cmap='gray')
                        ax[k,band].axis('off')
                    else:
                        ax[k].imshow(result[0,:,:,k],interpolation='nearest',cmap='gray')
                        ax[k].axis('off')

                # if type(self.nodes[node_to_plot])==S1_Layer:
                #     for k in range(num_filters_to_plot):

                #         ki = 4+k
                #         ax2[k,band].imshow(result[0,:,:,ki],interpolation='nearest',cmap='gray')
                #         ax2[k,band].axis('off')

            if type(self.nodes[node_to_plot])==S1_Layer:
                fig.savefig(os.path.join(node_output_dir,'output_phase0.pdf'))
                #fig2.savefig(os.path.join(node_output_dir,'output_phase1.pdf'))
                #plt.close(fig2)
            else:
                fig.savefig(os.path.join(node_output_dir,'output.pdf'))

            plt.close(fig)
