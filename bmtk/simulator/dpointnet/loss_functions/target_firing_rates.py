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

        '''
        if spikes.dtype != self._dtype:
            spikes = tf.cast(spikes, self._dtype)

        rates = tf.reduce_mean(spikes, (0, 1)) # calculate the mean firing rate over time and batch

        # reg_loss = loss_utils.compute_spike_rate_target_loss(rates, self._target_rates, dtype=self._dtype)

        for key, value in target_rates.items():
            neuron_ids = value["neuron_ids"]
            if len(neuron_ids) != 0:
                _rate_type = tf.gather(rates, neuron_ids)
                target_rate = value["sorted_target_rates"]
                # if core_mask is not None:
                #     key_core_mask = np.isin(value["neuron_ids"], core_neurons_ids)
                #     neuron_ids =  np.where(key_core_mask)[0]
                #     _rate_type = tf.gather(rates, neuron_ids)
                #     target_rate = value["sorted_target_rates"][key_core_mask]
                # else:
                #     _rate_type = tf.gather(rates, value["neuron_ids"])
                #     target_rate = value["sorted_target_rates"]

                loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
                total_loss += tf.reduce_sum(loss_type)
                num_neurons += tf.size(neuron_ids)

        total_loss /= tf.cast(num_neurons, dtype=dtype)

        return reg_loss * self._rate_cost
        '''

def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # Firstly we shuffle the current model rates to avoid bias towards a particular tuning angles (inherited from neurons ordering in the network)
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rates = tf.gather(_rates, rand_ind)
    sorted_rate = tf.sort(_rates)
    # u = target_rate - sorted_rate
    u = sorted_rate - target_rate
    n = tf.shape(target_rate)[0]
    tau = (tf.cast(tf.range(n), dtype) + 1) / tf.cast(n, dtype)
    loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype)

    return loss


def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    tau = tf.cast(tau, dtype)
    abs_u = tf.abs(u)
    num = tf.abs(tau - tf.cast(u <= 0, dtype))
    branch_1 = num / (2 * kappa) * tf.square(u)
    branch_2 = num * (abs_u - 0.5 * kappa)
    return tf.where(abs_u <= kappa, branch_1, branch_2)