# -*- coding: utf-8 -*-
import os, sys
from pprint import pprint
from neuron import h

from bmtk.simulator import bionet
from bmtk.simulator.bionet.io_tools import io


@bionet.model_processing
def add_channels(hoc_obj, cell, dynamics_params):
    for sec in hoc_obj.all:
        if 'axon' in sec.name():
            continue

        sec.insert('iahp')
        setattr(sec, 'gkbar_iahp', 3.0e-04)
        setattr(sec, 'taum_iahp', 0.5)
        
    return hoc_obj


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()
    bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.added_channels.json')
