import os, sys
import matplotlib.pyplot as plt

from bmtk.analyzer.compartment import plot_traces
from bmtk.analyzer.spike_trains import plot_raster


def plot_results(config_file):
    plot_raster(config_file=config_file, show=False, population='sources')
    plot_raster(config_file=config_file, show=False, population='targets')
    plot_traces(config_file=config_file, report_name='membrane_potential')
    plt.show()


if __name__ == '__main__':
    if os.path.basename(__file__) != sys.argv[-1]:
        plot_results(sys.argv[-1])
    else:
        plot_results('config.simulation_iclamp.json')