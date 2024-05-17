import os
import matplotlib.pyplot as plt

from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_raster

if os.path.exists('outputs/output_tdep'):
    plot_raster(config_file='config.comsol_tdep.json', show=False)
    plot_traces(config_file='config.comsol_tdep.json', report_name='membrane_potential', show=False)

if os.path.exists('outputs/output_stat'):
    plot_raster(config_file='config.comsol_stat.json', show=False)
    plot_traces(config_file='config.comsol_stat.json', report_name='membrane_potential', show=False)

if os.path.exists('outputs/output_stat2'):
    plot_raster(config_file='config.comsol_stat2.json', show=False)
    plot_traces(config_file='config.comsol_stat2.json', report_name='membrane_potential', show=False)

plt.show()
