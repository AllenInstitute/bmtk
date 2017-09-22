import sys
import os
import glob

import neuron
from neuron import h
from bmtk.simulator.bionet.pyfunction_cache import py_modules
from bmtk.simulator.bionet.pyfunction_cache import load_py_modules
from bmtk.simulator.bionet.pyfunction_cache import synapse_model, synaptic_weight, cell_model


pc = h.ParallelContext()



    
def quit_execution(): # quit the execution with a message

    pc.done()
    sys.exit()
    
    return


def clear_gids():
    pc.gid_clear()
    pc.barrier()
    
def load_neuron_modules(conf, **cm):

    h.load_file('stdgui.hoc')

    bionet_dir = os.path.dirname(__file__)
    h.load_file(bionet_dir+'/import3d.hoc') # loads hoc files from package directory ./import3d. It is used because read_swc.hoc is modified to suppress some warnings.
    h.load_file(bionet_dir+'/advance.hoc')

    neuron.load_mechanisms(str(conf["components"]["mechanisms"]))
    load_templates(conf["components"]["templates"])




def load_templates(template_dir):
    
    '''
    Load all templates to be available in the hoc namespace for instantiating cells
    '''
    cwd = os.getcwd()
    os.chdir(template_dir)

    hoc_templates = glob.glob("*.hoc")

    for hoc_template in hoc_templates:
        h.load_file(str(hoc_template))

    os.chdir(cwd)



        