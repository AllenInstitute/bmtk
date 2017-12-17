from bmtk.analyzer.visualization.spikes import plot_spikes, plot_rates

#plot_spikes('network/V1_nodes.h5', 'network/V1_node_types.csv', 'output/spikes.txt', group_key='pop_name')

plot_rates('network/V1_nodes.h5', 'network/V1_node_types.csv', 'output/spikes.txt', group_key='pop_name', smoothed=True)