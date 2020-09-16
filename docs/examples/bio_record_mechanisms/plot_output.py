import matplotlib.pyplot as plt

from bmtk.analyzer.spike_trains import plot_raster, plot_rates, plot_rates_boxplot
from bmtk.analyzer.compartment import plot_traces


plot_raster(config_file='config.json', group_by='pop_name', show=False)
plot_rates(config_file='config.json', show=False)

plot_traces(config_file='config.json', node_ids=range(10), report_name='calcium_concentration', show=False)
plot_traces(config_file='config.json',  node_ids=range(10), report_name='membrane_potential', show=False)


plt.show()
