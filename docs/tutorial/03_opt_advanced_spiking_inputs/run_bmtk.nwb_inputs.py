import sys
from bmtk.simulator import bionet


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    net = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=net)
    sim.run()

    bionet.nrn.quit_execution()


if __name__ == '__main__':
    # Find the appropriate config.json file
    if __file__ != sys.argv[-1]:
        config_path = sys.argv[-1]
        run(config_path)
    else:
        run('config.nwb_inputs.json')
        # run('config.simulation.units_map.json')
        # run('config.simulation.multi_session.json')       
