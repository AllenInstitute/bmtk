import matplotlib.pyplot as plt

from bmtk.analyzer.spike_trains import plot_raster, plot_rates


plot_raster(config_file='config.simulation.json', title='Raster', show=False)
plot_rates(config_file='config.simulation.json', title='Rates', show=False)
plt.show()
