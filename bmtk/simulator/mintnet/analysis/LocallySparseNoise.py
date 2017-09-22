import numpy as np
import pandas as pd
import h5py


class LocallySparseNoise (object):

    def __init__(self,data_file_name):

        self.stim_table = pd.read_hdf(data_file_name,'stim_table')
        self.node_table = pd.read_hdf(data_file_name,'node_table')


        self.data_file_name = data_file_name

        data = h5py.File(self.data_file_name,'r')

        self.data_sets = data.keys()
        self.data_sets.remove('stim_table')
        self.data_sets.remove('node_table')
        self.data_sets.remove('stim_template')

        self.stim_template = data['stim_template'].value

        data.close()

    @staticmethod
    def rf(response, stim_template, stim_shape):
        T = stim_template.shape[0]
        rf_shape = tuple(stim_template.shape[1:])

        unit_shape = tuple(response.shape[1:])

        response.resize([T,np.prod(unit_shape)])

        rf = np.dot(response.T,stim_template)

        rf_new_shape = tuple([rf.shape[0]] + list(rf_shape))
        rf.resize(rf_new_shape)
        rf_final_shape = tuple(list(unit_shape) + list(stim_shape))
        rf.resize(rf_final_shape)

        return rf

    def compute_receptive_fields(self, dtype=np.float32):

        output = h5py.File(self.data_file_name[:-3]+'_analysis.ic','a')
        data = h5py.File(self.data_file_name,'r')

        # convert to +/-1 or 0
        stim_template = data['stim_template'].value.astype(dtype)
        stim_template = stim_template-127
        stim_template = np.sign(stim_template)
        #print np.unique(stim_template)

        stim_shape = tuple(stim_template.shape[1:])
        T = stim_template.shape[0]

        stim_template.resize([T,np.prod(stim_shape)])

        stim_template_on = stim_template.copy()
        stim_template_off = stim_template.copy()

        stim_template_on[stim_template_on<0] = 0.0
        stim_template_off[stim_template_off>0] = 0.0

        for data_set in self.data_sets:

            response = data[data_set].value
            response = response - np.mean(response,axis=0)

            key_onoff = data_set+'/lsn/on_off'
            key_on = data_set+'/lsn/on'
            key_off = data_set+'/lsn/off'
            for key in [key_onoff, key_on, key_off]:
                if key in output:
                    del output[key]

            output[key_onoff] = self.rf(response, stim_template, stim_shape)
            output[key_on] = self.rf(response, stim_template_on, stim_shape)
            output[key_off] = self.rf(response, stim_template_off, stim_shape)

        data.close()
