import matplotlib.pyplot as plt

from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot

config_file = 'config.simulation.json'
# config_file = 'config.simulation_vm.json'
# config_file = 'config.simulation_ecp.json'

# Setting show to False so we can display all the plots at the same time
plot_raster(config_file=config_file, show=False) # , group_by='model_name', show=False)
plot_rates_boxplot(config_file=config_file, group_by='model_name', show=False)
if config_file == 'config.simulation_vm.json':
    plot_traces(
        config_file='config.simulation_vm.json', report_name='membrane_potential', group_by='model_name',
        times=(0.0, 200.0), show=False
    )

plt.show()
