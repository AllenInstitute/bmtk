import matplotlib.pyplot as plt
import sys
sys.path.append('../../..')
sys.path.append('../..')
sys.path.append('..')
sys.path.append('../bio_components')
from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_raster, plot_rates_boxplot
import bio_components.plot as bioplot

plot_raster(config_file='config_comsol_1c.json')

plot_raster(config_file='config_comsol_2x.json')

bioplot.plot_activity_3d(
    nodes_dir = 'network/column_nodes.h5',
    electrodes_dir = '../bio_components/stimulations/1c.csv',
    spikes_dir = 'outputs/output_1c/spikes.csv',
    save_dir = 'figures/1c'
)

bioplot.plot_activity_3d(
    nodes_dir = 'network/column_nodes.h5',
    electrodes_dir = '../bio_components/stimulations/2x.csv',
    spikes_dir = 'outputs/output_2x/spikes.csv',
    save_dir = 'figures/2x'
)