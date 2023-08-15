import matplotlib.pyplot as plt

from bmtk.analyzer.spike_trains import plot_raster, plot_rates, plot_rates_boxplot


# Setting show to False so we can display all the plots at the same time
plot_raster(config_file='config.simulation.json', group_by='model_name', show=False)
plot_rates(config_file='config.simulation.json', group_by='model_name', smoothing=True, show=False)
plot_rates_boxplot(config_file='config.simulation.json', group_by='model_name', show=False)

plt.show()
