from bmtk.analyzer.visualization.spikes import plot_raster, plot_rates
import matplotlib.pyplot as plt

plt.figure('Raster')
plot_raster('output/spikes.h5', with_histogram=True, with_labels=['v1'],
            nodes_file='network/internal_nodes.h5',
            node_types_file='network/internal_node_types.csv',
            group_by='model_name',
            show_plot=False)
plt.figure('Rates')
plot_rates('output/spikes.h5',
           nodes_file='network/internal_nodes.h5',
           node_types_file='network/internal_node_types.csv',
           group_by='model_name',
           show_plot=False)
plt.show()
