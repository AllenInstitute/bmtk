from neuron import h
from bmtk.simulator.bionet.cell import Cell


pc = h.ParallelContext()    # object to access MPI methods


class PointSomaCell(Cell):
    """Used to represent single compartment cells with neural mechanisms"""
    def __init__(self):
        # TODO: Implement
        raise NotImplementedError('Point Soma cell types are not currently implemented.')
