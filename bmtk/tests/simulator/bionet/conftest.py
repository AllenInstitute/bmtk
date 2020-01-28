try:
    import bmtk.simulator.bionet as bionet
    from bmtk.simulator.bionet.gids import GidPool
    from bmtk.simulator.bionet.pyfunction_cache import *

    nrn_installed = True

except ImportError:
    nrn_installed = False