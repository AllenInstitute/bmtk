import matplotlib.pyplot as plt

from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_raster

plot_raster(config_file='config.comsol_tdep.json', show=False)
plot_traces(config_file='config.comsol_tdep.json', report_name='membrane_potential', show=False)

plot_raster(config_file='config.comsol_stat.json', show=False)
plot_traces(config_file='config.comsol_stat.json', report_name='membrane_potential', show=False)

plot_raster(config_file='config.comsol_stat2.json', show=False)
plot_traces(config_file='config.comsol_stat2.json', report_name='membrane_potential', show=False)

plt.show()
