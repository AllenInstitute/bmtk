import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import numbers
from enum import Enum

from bmtk.simulator import dpointnet
from bmtk.simulator.dpointnet import register_loss_module
from bmtk.simulator.dpointnet.id_maps import TFIDMap
from bmtk.simulator.dpointnet.io_tools import io


class RegressionMethod(Enum):
    mae = 0
    rmse = 1
    huber = 2


@register_loss_module(module_name='TargetFiringRate')
class CustomTargetFiringRate:
    """Custom Loss function for regulating firing rates"""
    def __init__(self, rnn, firing_rate, **kwargs):
        self.rnn = rnn
        self.args = kwargs
        self.args['firing_rate'] = firing_rate

        if isinstance(firing_rate, str):
            self.target_tf_ids, self.target_frs = self._load_targets_from_csv(firing_rate, **kwargs)
        elif isinstance(firing_rate, (numbers.Number, np.number)):
            self.target_tf_ids = None  # Setting it to None should encapsulate all neurons
            self.target_frs = firing_rate * self.rnn.dt / 1000.0
        else:
            raise NotImplementedError()
        
        self.one_over_n = 1.0/float(len(self.target_tf_ids))

        self._pre_delay = kwargs.get('pre_delay', 0.0)
        self._post_delay = kwargs.get('post_delay', 0.0)
        self._trim_spikes = self._pre_delay > 0.0 or self._post_delay > 0.0
        if self._trim_spikes:
            # pre/post delay is passed in as milliseconds (float). Change it to the number of timesteps
            # (int) to cut from the begging/end of spikes table.
            self._pre_delay = int(self._pre_delay/self.rnn.dt)
            self._post_delay = None if self._post_delay == 0 else -int(self._post_delay/self.rnn.dt)

        method_name = kwargs.get('method', 'mae')
        try:
            self._method = RegressionMethod[method_name]
        except KeyError as ke:
            raise KeyError(f'{self.__class__.__name__}: Unknown method {method_name}. Available options: {", ".join([m.name for m in RegressionMethod])}')

        if self._method == RegressionMethod.huber:
            self._huber_delta = kwargs.get('huber_delta', None)
            if self._huber_delta is None:
                raise ValueError(f'{self.__class__.__name__}: When using method="huber" please specify a "huber_delta" value.')

    def _find_col(self, rates_df, col_name_attr, aliases, default=None):
        column_name = self.args.get(col_name_attr, None)
        if column_name is not None:
            if column_name not in rates_df.columns:
                csv_path = self.args.get('firing_rate', 'firing_rate csv file')
                raise ValueError(f'{self.__class__.__name__}: Could not find column {column_name} in {csv_path}.')
            return column_name
        
        for c in aliases:
            if c in rates_df.columns:
                return c
        return default

    def _load_targets_from_csv(self, csv_path, **kwargs):
        sep = kwargs.get('sep', ',')
        rates_df = pd.read_csv(csv_path, sep=sep)
        target_rates_col = self._find_col(rates_df, kwargs.get('firing_rate_col', None), ['firing_rate', 'firing_rates', 'rates', 'fr'])
        # target_rates_col = kwargs.get('firing_rate_col', self._find_col(rates_df, ['firing_rate', 'firing_rates', 'rates', 'fr']))
        if target_rates_col is None:
            raise ValueError(f'Unable to finding "firing_rate" column in {csv_path}. If using custom csv please set "firing_rate_col" value to appropiate column.')

        groupby = kwargs.pop('groupby', None)
        if groupby is None or groupby == 'node_id' or 'node_id' in groupby:
            return self._load_csv_individual(rates_df=rates_df, rates_col=target_rates_col, **kwargs)
        else:
            return self._load_csv_grouped(rates_df=rates_df, rates_col=target_rates_col, groupby=groupby, **kwargs)

    def _load_csv_grouped(self, rates_df, rates_col, groupby, **kwargs):
        prop_names = [groupby] if isinstance(groupby, str) else groupby
        
        rec_network = self.rnn.get_recurrent_network()
        target_tfids = np.empty(0, dtype=int)
        target_frs = np.empty(0, dtype=float)
        for prop_vals, props_df in rates_df.groupby(groupby):
            rates = props_df[rates_col].values
            if len(rates) > 1:
                csv_path = self.args['firing_rate']
                raise ValueError(f'{self.__class__.__name__}: {csv_path} contains multiple rows for group "{prop_names}", only one target firing rate supported.'
                                 ' Please consider using SpikeRateDistributionTarget loss module.')
            firing_rate_target = rates[0]
            
            prop_vals = [prop_vals] if isinstance(prop_vals, str) else prop_vals
            query = {n: v for n, v in zip(prop_names, prop_vals)}
            tf_ids = rec_network.get_tf_ids(**query)
            if len(tf_ids) == 0:
                io.log_warning(f'Could not find any matching nodes for query "{query}')
                continue
            else:
                target_tfids = np.concatenate((target_tfids, tf_ids))
                target_frs = np.concatenate((target_frs, np.full(len(tf_ids), firing_rate_target)))

        return tf.constant(target_tfids), target_frs*self.rnn.dt/1000.0
        
    def _load_csv_individual(self, rates_df, rates_col, **kwargs):
        node_id_col = kwargs.get('node_id_col', None)
        if node_id_col is None:
            for c in ['node_id', 'node_ids', 'gid', 'gids']:
                if c in rates_df.columns:
                    node_id_col = c
                    break
            else:
                raise ValueError(f'Could not find valid node_id column in csv file')
            
        pop_col = kwargs.get('population_col', None)
        if pop_col is None:
            for c in ['population', 'populations']:
                if c in rates_df.columns:
                    pop_col = c
                    break
        
        recurrent_ids = TFIDMap().recurrent_bmtk_ids()
        if pop_col is None:
            if len(recurrent_ids.keys()) == 1:
                pop_name = list(recurrent_ids.keys())[0]
                pop_col = 'population'
                rates_df[pop_col] = pop_name
            else:
                raise ValueError(f'Multiple recurrent node populations, please specify "population" column.')
        
        firing_rate_col = kwargs.get('firing_rate_col', None)
        if firing_rate_col is None:
            for c in ['firing_rate', 'firing_rates', 'rates', 'fr']:
                if c in rates_df.columns:
                    firing_rate_col = c
                    break
            else:
                raise ValueError(f'Unable to finding "firing_rate" column in csv file. If using custom csv please set "firing_rate_col" value to appropiate column.')

        n_rows = len(rates_df)
        target_tfids = np.zeros(n_rows, dtype=int)
        target_frs = np.zeros(n_rows, dtype=float)
        idx_beg, idx_end = 0, 0
        for pop_name, pop_subdf in rates_df.groupby(pop_col):
            idx_end = idx_beg + len(pop_subdf)
            target_frs[idx_beg:idx_end] = pop_subdf[firing_rate_col].values
            node_ids = pop_subdf[node_id_col].values
            target_tfids[idx_beg:idx_end] = recurrent_ids[pop_name][node_ids]           
            idx_beg = idx_end

        if idx_end < n_rows:
            target_frs = target_frs[:idx_end]
            target_tfids = target_tfids[:idx_end]
        
        target_frs = target_frs*self.rnn.dt/1000.0 # *self.rnn.dt/1000)

        return target_tfids, target_frs

    def __call__(self, spikes, **kwargs):
        if self.target_tf_ids is not None:
            # Get only the tf-ids used to calculate the target firing rates; in some cases only a subset may be used, or their order
            # may be different
            spikes = tf.gather(spikes, indices=self.target_tf_ids, axis=2)
            
        if self._trim_spikes:
            # If there is pre/post delay trim the spikes
            spikes = spikes[:, self._pre_delay:self._post_delay, :]

        actual_rates = tf.reduce_mean(spikes, (0, 1))  # Actually gets the firing rate per unit of time (in most cases milliseconds)
        rates_diff = actual_rates - self.target_frs

        if self._method == RegressionMethod.mae:
            abs_err = tf.abs(rates_diff)
            loss = tf.reduce_sum(abs_err)*self.one_over_n
        
        elif self._method == RegressionMethod.rmse:
            ms_err = tf.reduce_sum(tf.square(rates_diff))*self.one_over_n
            loss = tf.sqrt(ms_err)

        elif self._method == RegressionMethod.huber:
            abs_err = tf.abs(rates_diff)
            loss_small = 0.5*tf.square(abs_err)
            loss_large = self._huber_delta*(abs_err - 0.5*self._huber_delta)
            vals = tf.where(abs_err <= self._huber_delta, loss_small, loss_large)
            loss = tf.reduce_sum(vals)*self.one_over_n
        else:
            # Should never get here.
            return RuntimeError()
        
        return loss


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
