import sys, os

from bmtk.simulator import bionet


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    net = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=net)
    sim.run()
    bionet.nrn.quit_execution()

    from bmtk.analyzer.visualization.spikes import plot_spikes, plot_rates
    plot_spikes('network/nodes.h5', 'network/node_types.csv', 'output/spikes.h5', save_as = 'rasters/plot.png',group_key='model_name')

if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config_A.json')
        # run('config_je1e80g3jext5e-4_10in50.json')

