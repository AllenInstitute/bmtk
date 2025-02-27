import sys
from bmtk.simulator import bionet
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet import spikes_generator
import numpy as np


@spikes_generator
def my_spikes_generator(node, sim):
    io.log_info(f'Generating custom spike trains for {node.node_id} from {node.population_name}')
    if node['pop_name'] == 'tON':
        return np.arange(100.0, sim.tstop, step=sim.dt*10)
    elif node['pop_name'] == 'tOFF':
        return np.arange(100.0, sim.tstop, step=sim.dt*20)
    else:
        return []

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
        run('config.spikes_generator.json')
   
