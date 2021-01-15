import os, sys
from bmtk.simulator import bionet
from bmtk.analyzer.spike_trains import plot_raster


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()



    # Plot the spike trains raster
    plot_raster(config_file='config.bionet.json', group_by='pop_name')
    # bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('simulation_config.json')
