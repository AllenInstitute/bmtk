import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

from bmtk.simulator import popnet


fig_size = (5.0, 5.7)
legend_loc = 'upper right'
legend_alpha = 0.95

def plot_rates(schema, zoom=True):
    spikes_df = pd.read_csv('output_{}/spike_rates.txt'.format(schema), sep=' ',
                            names=['population', 'times', 'rates'])

    spikes_df['times'] *= 1000.0

    plt.figure(figsize=fig_size)
    if zoom:
        spikes_df = spikes_df[(spikes_df['times'] >= 1000.0) & (spikes_df['times'] <= 1200.0)]

    rates_exc = spikes_df[spikes_df['population'] == 100]
    rates_inh = spikes_df[spikes_df['population'] == 101]

    plt.plot(rates_exc['times'], rates_exc['rates'], color='blue', label='excitatory')
    plt.plot(rates_inh['times'], rates_inh['rates'], color='red', label='inhibitory')
    plt.xlabel('time (ms)', fontsize=18, weight='normal')
    plt.ylabel('firing rate (Hz)', fontsize=18, weight='normal')
    plt.legend(loc=legend_loc, framealpha=legend_alpha, prop={'size': 12})
    plt.xlim([1000, 1200])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tick_params(which='major', labelsize=12)
    plt.tight_layout()

    #plt.savefig('brunel_{}_popnet.png'.format(schema))
    #plt.savefig('brunel_{}_popnet.eps'.format(schema), format='eps', dpi=1000)
    plt.show()



def main(config_file):
    config_file = 'config.json'
    # config_file = 'config_' + sim+ '.json'
    configure = popnet.config.from_json(config_file)
    network = popnet.PopNetwork.from_config(configure)
    sim = popnet.PopSimulator.from_config(configure, network)
    sim.run()

    # cells_file = 'network/brunel_node_types.csv'
    # rates_file = configure['output']['rates_file']
    # plot_rates(schema='D_v2')
    # plot_rates_popnet(cells_file, rates_file, model_keys='pop_name')


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        print('Please give A,B,C or D')
