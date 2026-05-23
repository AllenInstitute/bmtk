import tensorflow as tf


class TargetFiringRate:
    def __init__(self, rnn, firing_rate, **kwargs):
        self.rnn = rnn
        self.target_fr = firing_rate
        # self.target_fr2 = tf.math.square(firing_rate)
        self.dt = self.rnn.dt
        self.seq_len = self.rnn.seq_len
        self.sim_time = self.dt*self.seq_len/1000.0
        # self.batch_size = self.rnn.batch_size

    @staticmethod
    def module():
        return 'TargetFiringRate'
    
    def __call__(self, spikes, **kwargs):
        n_neurons = spikes.shape[-1]
        
        actual_fr = tf.reduce_mean(spikes)/n_neurons/self.sim_time/self.rnn.batch_size
        diff_fr = actual_fr - self.target_fr
        diff_fr2 = tf.math.square(diff_fr)
        mse =  tf.math.square(diff_fr2)
        # print('mse = ', mse)
        return mse