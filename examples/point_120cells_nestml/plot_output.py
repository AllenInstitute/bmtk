import matplotlib.pyplot as plt

from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot


# Setting show to False so we can display all the plots at the same time
plot_raster(config_file='config.simulation.json', group_by='pop_name', show=False)
plot_rates_boxplot(config_file='config.simulation.json', group_by='pop_name', show=False)
plot_traces(
    config_file='config.simulation.json', report_name='membrane_potential', group_by='pop_name',
    times=(0.0, 200.0), show=False
)

plt.show()
