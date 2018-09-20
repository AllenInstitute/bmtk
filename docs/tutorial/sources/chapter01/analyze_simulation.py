from bmtk.analyzer.spike_trains import to_dataframe
from bmtk.analyzer.cell_vars import plot_report

config_file = 'simulation_config.json'
print to_dataframe(config_file=config_file)
plot_report(config_file=config_file)
