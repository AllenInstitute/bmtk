import sys
from bmtk.simulator import filternet
from bmtk.simulator.filternet.pyfunction_cache import load_py_modules


def run(config_file):
    config = filternet.Config.from_json(config_file)
    config.build_env()
    # load_py_modules(cell_processors=cell_loaders)

    net = filternet.FilterNetwork.from_config(config)
    sim = filternet.FilterSimulator.from_config(config, net)
    sim.run()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')