import matplotlib.pyplot as plt

from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot


plot_raster(config_file='sim_ch04/simulation_config.json', group_by='pop_name', show=False)
plot_rates_boxplot(config_file='sim_ch04/simulation_config.json', group_by='pop_name', show=False)
plt.show()