import matplotlib.pyplot as plt
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot


# fig = plot_raster(spikes_file='output/spikes.h5', show=False) # , group_by='model_name', show=False)
fig = plot_raster(config_file='config.simulation.json', group_by='model_template', show=False)

# fig.savefig('my_raster.png')
plt.show()