import matplotlib.pyplot as plt
from bmtk.analyzer.spike_trains import plot_raster, plot_rates
from bmtk.analyzer.compartment import plot_traces


plot_raster(config_file='config.simulation.json', group_by='pop_name', show=False)
plot_rates(config_file='config.simulation.json', show=False)
plot_traces(config_file='config.simulation.json', node_ids=range(10), report_name='calcium_concentration', show=False)
plot_traces(config_file='config.simulation.json',  node_ids=range(10), report_name='membrane_potential', show=False)

plt.show()
