import pytest
import tempfile
import numpy as np


from bmtk.builder.builder_utils import check_properties_across_ranks
from bmtk.builder import NetworkBuilder

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
    has_mpi = True
except:
    MPI_rank = 0
    MPI_size = 1
    has_mpi = False


def tmpdir():
    tmp_dir = tempfile.mkdtemp() if MPI_rank == 0 else None
    tmp_dir = comm.bcast(tmp_dir, 0)
    return tmp_dir


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_create_network_dir_mpi():
    # For old MPI issue were both ranks were trying to create the output_dir at the same time.
    tdir = tmpdir()
    net = NetworkBuilder('v1')
    net.add_nodes(N=1, model_type='biophysical')
    net.build()
    net.save_nodes(output_dir=tdir)


# check_properties_across_ranks
@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_check_properties_mpi():
    check_properties_across_ranks({
        'p1': 'p1',
        'p2': 2.22,
        'p3': 3,
        'p4': True,
        'p5': [0, 1, 2, 3],
        'p6': ['a', 'b', 'c', 'd'],
        'p7': np.array([1.0, 2.0, 3.0, 4.0])
    })


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_check_nonuniform_args():
    # Check that the number of arguments is the same across ranks
    with pytest.raises(IndexError):
        props = {
            'p1': 1,
            'p2': 'two'
        }
        if MPI_rank > 0:
            props['p3'] = 3.0
        check_properties_across_ranks(props)

    # Check the argument keys are the same
    with pytest.raises(TypeError):
        check_properties_across_ranks({
            'p{}'.format(MPI_rank): 0
        })


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_nonuniform_str_mpi():
    with pytest.raises(TypeError):
        check_properties_across_ranks({
            'p1': str(MPI_rank),
            'p2': 2.22,
            'p3': 3,
            'p4': True,
            'p5': [0, 1, 2, 3],
            'p6': ['a', 'b', 'c', 'd'],
            'p7': np.array([1.0, 2.0, 3.0, 4.0])
        })


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_nonuniform_int_mpi():
    with pytest.raises(Exception):
        check_properties_across_ranks({
            'p1': 'p1',
            'p2': 2.22,
            'p3': int(MPI_rank),
            'p4': True,
            'p5': [0, 1, 2, 3],
            'p6': ['a', 'b', 'c', 'd'],
            'p7': np.array([1.0, 2.0, 3.0, 4.0])
        })


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_nonuniform_float_mpi():
    with pytest.raises(TypeError):
        check_properties_across_ranks({
            'p1': 'p1',
            'p2': float(MPI_rank),
            'p3': 3,
            'p4': True,
            'p5': [0, 1, 2, 3],
            'p6': ['a', 'b', 'c', 'd'],
            'p7': np.array([1.0, 2.0, 3.0, 4.0])
        })


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_nonuniform_list_mpi():
    with pytest.raises(TypeError):
        check_properties_across_ranks({
            'p1': 'p1',
            'p2': 2.22,
            'p3': 3,
            'p4': True,
            'p5': [0, 1, 2, 3 + MPI_rank],
            'p6': ['a', 'b', 'c', 'd'],
            'p7': np.array([1.0, 2.0, 3.0, 4.0])
        })


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
def test_nonuniform_array_mpi():
    with pytest.raises(TypeError):
        check_properties_across_ranks({
            'p1': 'p1',
            'p2': float(MPI_rank),
            'p3': 3,
            'p4': True,
            'p5': [0, 1, 2, 3 + MPI_rank],
            'p6': ['a', 'b', 'c', 'd'],
            'p7': np.array([1.0, 2.0, 3.0, 4.0 + MPI_rank])
        })


if __name__ == '__main__':
    test_check_properties_mpi()
    test_check_nonuniform_args()
    test_nonuniform_str_mpi()
    test_nonuniform_float_mpi()
    test_nonuniform_int_mpi()
    test_nonuniform_list_mpi()
    test_nonuniform_array_mpi()
