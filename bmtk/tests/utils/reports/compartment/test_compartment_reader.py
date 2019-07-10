import os
import numpy as np

from bmtk.utils.reports import CompartmentReport

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhosts = comm.Get_size()
    barrier = comm.Barrier

except Exception as exc:
    rank = 0
    nhosts = 1
    barrier = lambda: None

cpath = os.path.dirname(os.path.realpath(__file__))
output_file = os.path.join(cpath, 'compartment_files/multi_population_report.h5')


def build_file():
    rank_cells = [(0, 10, 'v1'), (1, 50, 'v1'), (2, 100, 'v1'), (3, 1, 'v1'), (4, 200, 'v1'), (0, 100, 'v2'), (1, 50, 'v2')]
    cr = CompartmentReport(output_file, mode='w', tstart=0.0, tstop=100.0, dt=0.1, variable='Vm', units='mV')
    for node_id, n_elements, pop in rank_cells:
        cr.add_cell(node_id=node_id, population=pop, element_ids=np.arange(n_elements),
                    element_pos=np.zeros(n_elements))

    for i in range(1000):
        for node_id, n_elements, pop in rank_cells:
            cr.record_cell(node_id, population=pop, vals=[node_id]*n_elements, tstep=i)
    cr.close()


def test_compartment_reader():
    report = CompartmentReport(output_file, 'r')
    assert(len(report.populations) == 2)
    # Check v1 population
    assert('v1' in report.populations)
    v1_grp = report['v1']
    assert(np.all(np.sort(v1_grp.node_ids()) == np.arange(5)))
    assert(v1_grp.tstart() == 0.0)
    assert(v1_grp.tstop() == 100.0)
    assert(v1_grp.dt() == 0.1)
    assert(v1_grp.units() == 'mV')
    assert(v1_grp.n_elements() == 361)
    assert(v1_grp.element_pos().size == 361)
    assert(v1_grp.element_ids().size == 361)

    assert(v1_grp.data().shape == (1000, 361))
    assert(v1_grp.data(0).shape == (1000, 10))
    assert(v1_grp.data(0, time_window=(0.0, 50.0)).shape == (500, 10))
    assert(np.all(np.unique(v1_grp.data(0)) == [0.0]))

    assert(v1_grp.data(1).shape == (1000, 50))
    assert(np.all(np.unique(v1_grp.data(1)) == [1.0]))

    assert(v1_grp.data(2).shape == (1000, 100))
    assert(np.all(np.unique(v1_grp.data(2)) == [2.0]))

    assert(v1_grp.data(3).shape == (1000, 1))
    assert(np.all(np.unique(v1_grp.data(3)) == [3.0]))

    assert(v1_grp.data(4).shape == (1000, 200))
    assert(np.all(np.unique(v1_grp.data(4)) == [4.0]))

    # Check v2 population
    assert('v2' in report.populations)
    v1_grp = report['v2']
    assert(np.all(np.sort(v1_grp.node_ids()) == np.arange(2)))
    assert(v1_grp.tstart() == 0.0)
    assert(v1_grp.tstop() == 100.0)
    assert(v1_grp.dt() == 0.1)
    assert(v1_grp.units() == 'mV')
    assert(v1_grp.n_elements() == 150)
    assert(v1_grp.element_pos().size == 150)
    assert(v1_grp.element_ids().size == 150)

    assert(v1_grp.data().shape == (1000, 150))
    assert(v1_grp.data(0).shape == (1000, 100))
    assert(v1_grp.data(0, time_window=(0.0, 50.0)).shape == (500, 100))
    assert(np.all(np.unique(v1_grp.data(0)) == [0.0]))

    assert(v1_grp.data(1).shape == (1000, 50))
    assert(np.all(np.unique(v1_grp.data(1)) == [1.0]))


def test_compartment_reader2():
    report = CompartmentReport(output_file, 'r', default_population='v1')
    assert(len(report.populations) == 2)
    assert('v1' in report.populations)
    assert(np.all(np.sort(report.node_ids()) == np.arange(5)))
    assert(report.tstart() == 0.0)
    assert(report.tstop() == 100.0)
    assert(report.dt() == 0.1)
    assert(report.units() == 'mV')
    assert(report.n_elements() == 361)
    assert(report.element_pos().size == 361)
    assert(report.element_ids().size == 361)

    assert(report.data().shape == (1000, 361))
    assert(report.data(0).shape == (1000, 10))
    assert(report.data(0, time_window=(0.0, 50.0)).shape == (500, 10))
    assert(np.all(np.unique(report.data(0)) == [0.0]))

    assert(report.data(1).shape == (1000, 50))
    assert(np.all(np.unique(report.data(1)) == [1.0]))

    assert(report.data(2).shape == (1000, 100))
    assert(np.all(np.unique(report.data(2)) == [2.0]))

    assert(report.data(3).shape == (1000, 1))
    assert(np.all(np.unique(report.data(3)) == [3.0]))

    assert(report.data(4).shape == (1000, 200))
    assert(np.all(np.unique(report.data(4)) == [4.0]))

    # Check v2 population
    assert('v2' in report.populations)
    assert(np.all(np.sort(report.node_ids(population='v2')) == np.arange(2)))
    assert(report.tstart(population='v2') == 0.0)
    assert(report.tstop(population='v2') == 100.0)
    assert(report.dt(population='v2') == 0.1)
    assert(report.units(population='v2') == 'mV')
    assert(report.n_elements(population='v2') == 150)
    assert(report.element_pos(population='v2').size == 150)
    assert(report.element_ids(population='v2').size == 150)

    assert(report.data(population='v2').shape == (1000, 150))
    assert(report.data(0, population='v2').shape == (1000, 100))
    assert(report.data(0, population='v2', time_window=(0.0, 50.0)).shape == (500, 100))
    assert(np.all(np.unique(report.data(0, population='v2')) == [0.0]))

    assert(report.data(1, population='v2').shape == (1000, 50))
    assert(np.all(np.unique(report.data(1, population='v2')) == [1.0]))

if __name__ == '__main__':
    #build_file()
    #test_compartment_reader()
    test_compartment_reader2()


