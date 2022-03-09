try:
    import bmtk.simulator.bionet as bionet
    from bmtk.simulator.bionet.gids import GidPool
    from bmtk.simulator.bionet.pyfunction_cache import *
    from neuron import h

    h.load_file('stdrun.hoc')

    nrn_installed = True

except ImportError:
    nrn_installed = False
    has_mechanism = False


if nrn_installed:
    try:
        vecstim = h.VecStim()
        has_mechanism = True

    except AttributeError:
        has_mechanism = False
