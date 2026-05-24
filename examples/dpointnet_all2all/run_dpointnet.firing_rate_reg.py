import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

from bmtk.simulator import dpointnet
from bmtk.simulator.dpointnet import register_loss_module


@register_loss_module(module_name='TargetFiringRate')
class CustomTargetFiringRate:
    """Custom Loss function for regulating firing rates"""
    def __init__(self, rnn, firing_rate, **kwargs):
        self.rnn = rnn
        self.target_fr = firing_rate
        # self.target_fr2 = tf.math.square(firing_rate)
        self.dt = self.rnn.dt
        self.seq_len = self.rnn.seq_len
        self.sim_time = self.dt*self.seq_len/1000.0
        # self.batch_size = self.rnn.batch_size
   
    def __call__(self, spikes, **kwargs):
        # n_neurons = spikes.shape[-1]
        
        # mean_spike_count = tf.reduce_mean(tf.reduce_sum(spikes, axis=1), axis=0)
        # actual_fr = mean_spike_count / self.sim_time
        spike_counts = tf.reduce_sum(spikes, axis=1)
        firing_rates = spike_counts / self.sim_time

        # actual_fr = tf.reduce_mean(spikes)/n_neurons/self.sim_time/self.rnn.batch_size
        # diff_fr = actual_fr - self.target_fr
        # diff_fr2 = tf.math.square(diff_fr)
        # mse =  tf.math.square(diff_fr2)
        # print(spikes)
        # print('mse = ', mse)
        return tf.reduce_mean(tf.square(self.target_fr - firing_rates))



def run(config_path):
    config = dpointnet.Config.from_json(config_path)
    config.build_env()

    # Load network, training, and inference parameters from config
    rnn_network = dpointnet.RNN.from_config(config)
    rnn_network.build()

    # Run an inference once before model weights have been trained, to see resulting spike
    # train and firing-rates.
    untrained_results = rnn_network.run_inference()
    untrained_fr = untrained_results.spikes.mean_firing_rate()
    fig = untrained_results.spikes.raster(batch_nums=[0, 10], show=False)
    fig.suptitle(f'Untrained results, firing_rate = {untrained_fr}')
    fig.tight_layout()

    # Run training regimen (as specified in the config)
    rnn_network.train()
    
    # Rerun inference (as specified in the config) to get resulting spikes raters + fr.
    trained_results = rnn_network.run_inference()
    trained_fr = trained_results.spikes.mean_firing_rate()
    fig1 = trained_results.spikes.raster(batch_nums=[0, 10], show=False)
    fig1.suptitle(f'Trained Results; firing-rate = {trained_fr}')
    fig1.tight_layout()
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        type=str, 
        nargs='?', 
        default='config.train.json'
    )

    args, _ = parser.parse_known_args()
    run(args.config_path)
