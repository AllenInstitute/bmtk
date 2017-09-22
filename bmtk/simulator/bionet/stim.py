"""
Created on Aug 22, 2016

@author: sergeyg
"""
from neuron import h

"""
Representation of a Virtual/External/Stim node.

TODO:
 * Rename to Virtual
 * This is the biggest bottleneck when loading a network, find a way to optimize.
"""


class Stim(object):
    def __init__(self, stim_prop, spike_train_dataset):

        self.stim_gid = stim_prop.node_id
        self.rand_streams = []
        self._spike_train_dataset = spike_train_dataset

        self.set_stim(stim_prop, self._spike_train_dataset)
        
    def set_stim(self, stim_prop, spike_train):
        #spike_trains_handle = input_prop["spike_trains_handle"]
        #self.spike_train = spike_trains_handle['%d/data' % self.stim_gid][:]

        self.train_vec = h.Vector(spike_train[:])
        vecstim = h.VecStim()
        vecstim.play(self.train_vec)
        
        self.hobj = vecstim
