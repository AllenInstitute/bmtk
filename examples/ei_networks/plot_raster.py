import sys
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


marker_type = '.'
marker_size = 18.0
fig_size = (5.0, 5.7)

with_legend = True
legend_loc = 'upper right'
legend_alpha = 0.95


def plot_spikes(simulator, zoom=True, rates_all=True):
    spikes_df = pd.read_csv('{}/output/spikes.csv'.format(simulator), sep=' ')
    print(spikes_df['node_ids'].max())
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all', gridspec_kw={'height_ratios': [3, 1]},
                             figsize=fig_size)
    if zoom:
        #print(len(np.array(spikes_df['timestamps'])))
        spikes_df = spikes_df[(spikes_df['timestamps'] >= 1000.0) & (spikes_df['timestamps'] <= 1200.0)]
        # all_ts = np.array(spikes_df['timestamps'])
        all_ts = np.array(spikes_df['timestamps']) if rates_all else np.array(spikes_df[spikes_df['node_ids'] < 10000]['timestamps'])

        spikes_df = spikes_df[(spikes_df['node_ids'] >= 9960) & (spikes_df['node_ids'] <= 10010)]
        time_lim = [1000, 1200]
    else:
        all_ts = np.array(spikes_df['timestamps'])
        time_lim = [0.0, 3000.0]

    spikes_exc = spikes_df[spikes_df['node_ids'] < 10000]
    spikes_inh = spikes_df[spikes_df['node_ids'] >= 10000]
    axes[0].scatter(spikes_exc['timestamps'], spikes_exc['node_ids'], color='blue', marker=marker_type, s=marker_size, label='excitatory')
    axes[0].scatter(spikes_inh['timestamps'], spikes_inh['node_ids'], color='red', marker=marker_type, s=marker_size, label='inhibitory')
    axes[0].set_ylabel('node', fontsize=16, weight='normal')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].legend(loc=legend_loc, markerscale=2.0, framealpha=legend_alpha, prop={'size': 12})
    axes[0].tick_params(which='major', labelsize=12)

    spike_counts, ts = np.histogram(all_ts, bins=2000)

    axes[1].bar(ts[:-1], spike_counts/(12500.0 if rates_all else 10000.0)/0.001)
    axes[1].set_xlabel('time (ms)', fontsize=18, weight='normal')
    axes[1].set_ylabel('rate (Hz)', fontsize=18, weight='normal')
    axes[1].set_xlim(time_lim)
    axes[1].spines['right'].set_visible(False)

    axes[1].set_xticks(ticks=[1000, 1050, 1100, 1150, 1200])
    axes[1].tick_params(which='major', labelsize=12)

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.align_ylabels(axes)

    plt.savefig('brunel_{}.png'.format(simulator))
    plt.savefig('brunel_{}.eps'.format(simulator), format='eps', dpi=1000)
    plt.show()


def plot_rates(zoom=True):
    spikes_df = pd.read_csv('brunel_popnet/output_{}/spike_rates.txt'.format(schema), sep=' ',
                            names=['population', 'times', 'rates'])

    spikes_df['times'] *= 1000.0

    plt.figure(figsize=fig_size)
    if zoom:
        spikes_df = spikes_df[(spikes_df['times'] >= 1000.0) & (spikes_df['times'] <= 1200.0)]

    rates_exc = spikes_df[spikes_df['population'] == 100]
    rates_inh = spikes_df[spikes_df['population'] == 101]

    plt.plot(rates_exc['times'], rates_exc['rates'], color='blue', label='excitatory')
    # plt.plot(rates_inh['times'], rates_inh['rates'], color='red', label='inhibitory')
    plt.xlabel('time (ms)', fontsize=18, weight='normal')
    plt.ylabel('firing rate (Hz)', fontsize=18, weight='normal')
    # plt.legend(loc=legend_loc, framealpha=legend_alpha, prop={'size': 12})
    plt.xlim([1000, 1200])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tick_params(which='major', labelsize=12)
    plt.tight_layout()

    plt.savefig('brunel_popnet.png'.format(schema))
    plt.savefig('brunel_popnet.eps'.format(schema), format='eps', dpi=1000)
    plt.show()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        simulators = sys.argv[1:]
    else:
        simulators = ['bionet', 'pointnet', 'popnet']

    for sim in simulators:
        if sim.lower() == 'popnet':
            plot_rates(zoom=False)
        else:
            plot_spikes(simulator=sim, zoom=False)
