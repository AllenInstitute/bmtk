import os

try:
    import bmtk.simulator.bionet as bionet
    from bmtk.simulator.bionet.nrn import load_neuron_modules
    from bmtk.simulator.bionet.gids import GidPool
    from bmtk.simulator.bionet.pyfunction_cache import *
    from neuron import h

    h.load_file('stdrun.hoc')

    nrn_installed = True


except ImportError:
    nrn_installed = False
    has_mechanism = False


CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MORPH_DIR = os.path.join(CURR_DIR, 'components/morphology')
MECHS_DIR = os.path.join(CURR_DIR, 'components/mechanisms')

if nrn_installed:
    try:
        load_neuron_modules(mechanisms_dir=MECHS_DIR, templates_dir=None, default_templates=False)
        vecstim = h.VecStim()
        has_mechanism = True

    except AttributeError:
        has_mechanism = False
