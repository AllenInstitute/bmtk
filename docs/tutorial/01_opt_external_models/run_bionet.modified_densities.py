# -*- coding: utf-8 -*-
import os, sys
from pprint import pprint
from neuron import h

from bmtk.simulator import bionet
from bmtk.simulator.bionet.io_tools import io


@bionet.model_processing
def adjust_densities(hoc_obj, cell, dynamics_params):
    n_segs = 0
    n_segs_updated = 0
    for sec in hoc_obj.apic:
        for seg in sec:
            n_segs += 1
            if h.distance(seg) > 1000:
                n_segs_updated += 1
                org_cond = getattr(seg, 'gbar_Ih')
                new_cond = org_cond*0.5
                setattr(seg, 'gbar_Ih', new_cond)

    print(f'{n_segs_updated} out of {n_segs} apical dendritic segments updated.')
    return hoc_obj


# @bionet.model_processing(name='adjust_densities')
# def check_mechanisms(hoc_obj, cell, dynamics_params):
#     for sec in hoc_obj.apic:
#         print(sec.name())
#         pprint(sec.psection())
#         break


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
        run('config.simulation.json')
