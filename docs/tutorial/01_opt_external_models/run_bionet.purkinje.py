"""Simulates an example network of 450 cells receiving two kinds of external input as defined in the configuration file"""
import os
import sys
from bmtk.simulator import bionet
from bmtk.simulator.bionet.io_tools import io

from Purkinje_morpho_1 import Purkinje_Morpho_1


@bionet.cell_model(directive='python:Purkinje_morph_1', model_type='biophysical')
def loadPurkinjeModel(cell, template_name, dynamics_params):
    io.log_info(f'Loading cell {cell.node_id}, tempalate {template_name}, with spines_on={cell["spines_on"]}')
    cell = Purkinje_Morpho_1(cell['spines_on'])
    return cell


def run(config_path):
    conf = bionet.Config.from_json(config_path, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()
    bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        config_path = sys.argv[-1]
        run(config_path)
    else:
        # run('config.simulation_syns.json')
        run('config.simulation_iclamp.json')

