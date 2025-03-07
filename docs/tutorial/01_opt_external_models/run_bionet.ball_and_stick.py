"""Simulates an example network of 450 cells receiving two kinds of external input as defined in the configuration file"""
import os
import sys
from bmtk.simulator import bionet
from bmtk.simulator.bionet.io_tools import io

from neuron import h


class BallAndStick:
    def __init__(self, cell):
        self._gid = cell['node_id']
        self._setup_morphology(cell)
        self._setup_biophysics(cell)

    def _setup_morphology(self, cell):
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.all = [self.soma, self.dend]
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157
        self.dend.L = 200
        self.dend.diam = 1
    
    def _setup_biophysics(self, cell):
        for sec in self.all:
            sec.Ra = 100    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')                                          
        for seg in self.soma:
            seg.hh.gnabar = cell['gnabar']  # Sodium conductance in S/cm2
            seg.hh.gkbar = cell['gkbar']  # Potassium conductance in S/cm2
            seg.hh.gl = cell['gl']    # Leak conductance in S/cm2
            seg.hh.el = cell['el']     # Reversal potential in mV
        # Insert passive current in the dendrite                       # <-- NEW
        self.dend.insert('pas')                                        # <-- NEW
        for seg in self.dend:                                          # <-- NEW
            seg.pas.g = cell['g_pas']  # Passive conductance in S/cm2          # <-- NEW
            seg.pas.e = cell['e_pas']    # Leak reversal potential mV            # <-- NEW 
    
    def __repr__(self):
        return 'BallAndStick[{}]'.format(self._gid)


@bionet.cell_model(directive='python:loadBAS', model_type='biophysical')
def loadPurkinjeModel(cell, template_name, dynamics_params):
    print('HERE')
    bas_cell = BallAndStick(cell)
    return bas_cell
    # soma = h.Section(name='soma', cell=self)
    # self.dend = h.Section(name='dend', cell=self)
    # exit()
    # io.log_info(f'Loading cell {cell.node_id}, tempalate {template_name}, with spines_on={cell["spines_on"]}')
    # cell = Purkinje_Morpho_1(cell['spines_on'])
    # return cell


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

