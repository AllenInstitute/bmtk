import os
import tempfile
import numpy as np
import h5py
import pytest
from collections import namedtuple

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


@pytest.mark.skipif(nhosts > 1, reason="does not work with mpi")
def test_one_compartment_report():
    population = 'p1'
    output_file = tempfile.mkstemp(suffix='h5')[1]

    cr = CompartmentReport(output_file, mode='w', default_population=population,
                           tstart=0.0, tstop=100.0, dt=0.1)
    cr.add_cell(node_id=0, element_ids=[0], element_pos=[0.0])
    for i in range(1000):
        cr.record_cell(0, [i/100.0], tstep=i)

    cr.close()

    report_h5 = h5py.File(output_file, 'r')
    report_grp = report_h5['/report/{}'.format(population)]
    assert('data' in report_grp)
    data_ds = report_grp['data'][()]
    assert(report_grp['data'].size == 1000)
    assert(np.isreal(data_ds.dtype))
    assert(data_ds[0] == 0.00)
    assert(data_ds[-1] == 9.99)


    assert('mapping' in report_grp)
    mapping_grp = report_grp['mapping']
    assert(all(mapping_grp['element_ids'][()] == [0]))
    assert(mapping_grp['element_pos'][()] == [0.0])
    assert(mapping_grp['index_pointer'][()].size == 2)
    assert(mapping_grp['node_ids'][()] == [0])
    assert(np.allclose(mapping_grp['time'][()], [0.0, 100.0, 0.1]))
    os.remove(output_file)


@pytest.mark.skipif(nhosts > 1, reason="does not work with mpi")
def test_multi_compartment_report():
    population = 'cortical'
    output_file = tempfile.mkstemp(suffix='h5')[1]
    n_elements = 50

    cr = CompartmentReport(output_file, mode='w', default_population=population,
                           tstart=0.0, tstop=100.0, dt=0.1)
    cr.add_cell(node_id=0, element_ids=np.arange(n_elements), element_pos=[0.5]*n_elements)
    cr.initialize()
    for i in range(1000):
        cr.record_cell(0, [i+j for j in range(n_elements)], tstep=i)

    cr.close()

    report_h5 = h5py.File(output_file, 'r')
    report_grp = report_h5['/report/{}'.format(population)]
    assert('data' in report_grp)
    data_ds = report_grp['data'][()]
    assert(report_grp['data'].shape == (1000, n_elements))
    assert(np.isreal(data_ds.dtype))
    assert(data_ds[0, 0] == 0.0)
    assert(data_ds[999, n_elements-1] == 999.0+n_elements-1)

    assert('mapping' in report_grp)
    mapping_grp = report_grp['mapping']
    assert(np.allclose(mapping_grp['element_ids'][()], np.arange(n_elements)))
    assert(np.allclose(mapping_grp['element_pos'][()], [0.5]*n_elements))
    assert(mapping_grp['index_pointer'][()].size == 2)
    assert(mapping_grp['node_ids'][()] == [0])
    assert(np.allclose(mapping_grp['time'][()], [0.0, 100.0, 0.1]))
    os.remove(output_file)


def test_multi_cell_report(buffer_size=0):
    cells = [(0, 10), (1, 50), (2, 100), (3, 1), (4, 200)]
    total_elements = sum(n_elements for _, n_elements in cells)
    rank_cells = [c for c in cells[rank::nhosts]]
    output_file = os.path.join(cpath, 'output/multi_compartment_report.h5')
    population = 'cortical'

    cr = CompartmentReport(output_file, mode='w', default_population=population,
                           tstart=0.0, tstop=100.0, dt=0.1, variable='mebrane_potential', units='mV',
                           buffer_size=buffer_size)
    for node_id, n_elements in rank_cells:
        cr.add_cell(node_id=node_id, element_ids=np.arange(n_elements), element_pos=np.zeros(n_elements))

    for i in range(1000):
        for node_id, n_elements in rank_cells:
            cr.record_cell(node_id, [node_id+i/1000.0]*n_elements, tstep=i)
    cr.close()

    if rank == 0:
        report_h5 = h5py.File(output_file, 'r')
        report_grp = report_h5['/report/{}'.format(population)]
        assert('data' in report_grp)
        data_ds = report_grp['data'][()]
        assert(report_grp['data'].shape == (1000, total_elements))
        assert(np.isreal(data_ds.dtype))

        assert('mapping' in report_grp)
        mapping_grp = report_grp['mapping']
        assert(mapping_grp['element_ids'].size == total_elements)
        assert(mapping_grp['element_pos'].size == total_elements)
        assert(mapping_grp['index_pointer'].size == 6)
        assert(np.all(np.sort(mapping_grp['node_ids'][()]) == np.arange(5)))
        assert(np.allclose(mapping_grp['time'][()], [0.0, 100.0, 0.1]))

        os.remove(output_file)
    barrier()


def test_multi_population_report():
    cells = [(0, 10, 'v1'), (1, 50, 'v1'), (2, 100, 'v1'), (3, 1, 'v1'), (4, 200, 'v1'), (0, 100, 'v2'), (1, 50, 'v2')]
    rank_cells = [c for c in cells[rank::nhosts]]
    output_file = os.path.join(cpath, 'output/multi_population_report.h5')

    cr = CompartmentReport(output_file, mode='w', tstart=0.0, tstop=100.0, dt=0.1, variable='Vm', units='mV')
    for node_id, n_elements, pop in rank_cells:
        cr.add_cell(node_id=node_id, population=pop, element_ids=np.arange(n_elements),
                    element_pos=np.zeros(n_elements))

    for i in range(1000):
        for node_id, n_elements, pop in rank_cells:
            cr.record_cell(node_id, population=pop, vals=[node_id+i/1000.0]*n_elements, tstep=i)
    cr.close()

    if rank == 0:
        report_h5 = h5py.File(output_file, 'r')
        report_grp = report_h5['/report/{}'.format('v1')]
        assert('data' in report_grp)
        data_ds = report_grp['data'][()]
        assert(report_grp['data'].shape == (1000, 361))
        assert(np.isreal(data_ds.dtype))

        assert('mapping' in report_grp)
        mapping_grp = report_grp['mapping']
        assert(mapping_grp['element_ids'].size == 361)
        assert(mapping_grp['element_pos'].size == 361)
        assert(mapping_grp['index_pointer'].size == 6)
        assert(np.all(np.sort(mapping_grp['node_ids'][()]) == np.arange(5)))
        assert(np.allclose(mapping_grp['time'][()], [0.0, 100.0, 0.1]))

        report_grp = report_h5['/report/{}'.format('v2')]
        assert('data' in report_grp)
        data_ds = report_grp['data'][()]
        assert(report_grp['data'].shape == (1000, 150))
        assert(np.isreal(data_ds.dtype))

        assert('mapping' in report_grp)
        mapping_grp = report_grp['mapping']
        assert(mapping_grp['element_ids'].size == 150)
        assert(mapping_grp['element_pos'].size == 150)
        assert(mapping_grp['index_pointer'].size == 3)
        assert(np.all(np.sort(mapping_grp['node_ids'][()]) == [0, 1]))
        assert(np.allclose(mapping_grp['time'][()], [0.0, 100.0, 0.1]))

        os.remove(output_file)
    barrier()


def test_block_record():
    cells = [(0, 10), (1, 50), (2, 100), (3, 1), (4, 200)]
    total_elements = sum(n_elements for _, n_elements in cells)
    rank_cells = [c for c in cells[rank::nhosts]]
    output_file = os.path.join(cpath, 'output/multi_compartment_report.h5')
    population = 'cortical'

    cr = CompartmentReport(output_file, mode='w', default_population=population,
                           tstart=0.0, tstop=100.0, dt=0.1, variable='mebrane_potential', units='mV')
    for node_id, n_elements in rank_cells:
        cr.add_cell(node_id=node_id, element_ids=np.arange(n_elements), element_pos=np.zeros(n_elements))

    for node_id, n_elements in rank_cells:
        cr.record_cell_block(node_id, np.full((1000, n_elements), fill_value=node_id+1), beg_step=0, end_step=1000)

    cr.close()

    if rank == 0:
        report_h5 = h5py.File(output_file, 'r')
        report_grp = report_h5['/report/{}'.format(population)]
        assert('data' in report_grp)
        data_ds = report_grp['data'][()]
        assert(report_grp['data'].shape == (1000, total_elements))
        assert(np.isreal(data_ds.dtype))

        assert('mapping' in report_grp)
        mapping_grp = report_grp['mapping']
        assert(mapping_grp['element_ids'].size == total_elements)
        assert(mapping_grp['element_pos'].size == total_elements)
        assert(mapping_grp['index_pointer'].size == 6)
        assert(np.all(np.sort(mapping_grp['node_ids'][()]) == np.arange(5)))
        assert(np.allclose(mapping_grp['time'][()], [0.0, 100.0, 0.1]))

        os.remove(output_file)
    barrier()


def test_custom_columns():
    cells = [(0, 10), (1, 50), (2, 100), (3, 1), (4, 200)]
    total_elements = sum(n_elements for _, n_elements in cells)
    rank_cells = [c for c in cells[rank::nhosts]]
    output_file = os.path.join(cpath, 'output/multi_compartment_report.h5')
    population = 'cortical'

    cr = CompartmentReport(output_file, mode='w', default_population=population,
                           tstart=0.0, tstop=100.0, dt=0.1, variable='mebrane_potential', units='mV')
    for node_id, n_elements in rank_cells:
        cr.add_cell(node_id=node_id, element_ids=np.arange(n_elements), element_pos=np.zeros(n_elements), synapses=[node_id*2]*n_elements)

    for i in range(1000):
        for node_id, n_elements in rank_cells:
            cr.record_cell(node_id, [node_id+i/1000.0]*n_elements, tstep=i)
    cr.close()

    if rank == 0:
        report_h5 = h5py.File(output_file, 'r')
        report_grp = report_h5['/report/{}'.format(population)]
        assert('mapping' in report_grp)
        mapping_grp = report_grp['mapping']
        assert(mapping_grp['element_ids'].size == total_elements)
        assert(mapping_grp['element_pos'].size == total_elements)
        assert(mapping_grp['index_pointer'].size == 6)
        assert(np.all(np.sort(mapping_grp['node_ids'][()]) == np.arange(5)))
        assert(np.allclose(mapping_grp['time'][()], [0.0, 100.0, 0.1]))

        assert('synapses' in mapping_grp.keys())
        assert(mapping_grp['synapses'][()].size == total_elements)
        os.remove(output_file)
    barrier()


if __name__ == '__main__':
    #test_one_compartment_report()
    #test_multi_compartment_report()
    test_multi_cell_report()
    test_multi_population_report()
    test_block_record()
    test_custom_columns()