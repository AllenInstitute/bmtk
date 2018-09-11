"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import sys
import matplotlib.pyplot as plt
import h5py
import numpy as np

from bmtk.simulator import bionet


def show_cell_var(conf, var_name):
    vars_h5 = h5py.File(conf.reports['membrane_potential']['file_name'], 'r')
    gids = np.array(vars_h5['/mapping/gids'])
    indx_ptrs = np.array(vars_h5['/mapping/index_pointer'])
    t_start = vars_h5['/mapping/time'][0]
    t_stop = vars_h5['/mapping/time'][1]
    dt = vars_h5['/mapping/time'][2]
    times = np.linspace(t_start, t_stop, int((t_stop - t_start)/dt))
    var_table = vars_h5[var_name]['data']
    for plot_num, (gid, indx) in enumerate(zip(gids, indx_ptrs)):
        plt.subplot(len(gids), 1, plot_num+1)
        voltages = np.array(var_table[:, indx])
        plt.plot(times, voltages, label='gid={}'.format(gid))
        plt.legend(loc='upper right')
        plt.ylabel('V (mV)')

    plt.xlabel('time (ms)')
    plt.show()


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()

    show_cell_var(conf, 'v')


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        # Make sure to run only one at a time
        run('config_iclamp.json')  # Current clamp stimulation
        # run('config_xstim.json')  # Extracellular electrode stimulation
        # run('config_spikes_input.json')  # Synaptic stimulation with external virtual cells
