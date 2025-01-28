import numpy as np
from timeit import default_timer as timer
from memory_profiler import memory_usage
import tempfile
import h5py
from mpi4py import MPI
import time
import sys

from bmtk.utils.reports.spike_trains.adaptors.sonata_adaptors import write_sonata_old, write_sonata
from bmtk.utils.reports.spike_trains.adaptors.csv_adaptors import write_csv, write_csv_old

from bmtk.utils.reports.spike_trains import SpikeTrains, sort_order, sort_order_lu
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from bmtk.utils.io.ioutils import bmtk_world_comm
from bmtk.utils.reports.spike_trains.spike_train_buffer import STMemoryBuffer, STMPIBuffer, STCSVBuffer, STCSVMPIBuffer, STCSVMPIBufferV2

#comm = bmtk_world_comm.comm
#rank = bmtk_world_comm.MPI_rank
#size = bmtk_world_comm.MPI_size
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


N = 500000#0
n_spikes_avg = 90
n_spikes_std = 40


def save_in_mem():
    np.random.seed(1000)
    st = SpikeTrains(adaptor=STMPIBuffer() if size > 1 else STMemoryBuffer())
    for i in range(rank, N, size):
        # if i % 1000 == 0:
        #     print('{} > {}'.format(rank, i))
        st.add_spikes(node_ids=i, population='v1',
                      timestamps=np.random.uniform(0.0, 3000.0, size=np.random.randint(n_spikes_avg-n_spikes_std,
                                                                                       n_spikes_avg+n_spikes_std)))

    n_spikes = st.n_spikes('v1')
    if rank == 0:
        # print('HERE')
        print('finished In Memory Version, saving {} spikes...'.format(n_spikes))
        sys.stdout.flush()

    # write_csv('check_out.csv', st, sort_order=sort_order.by_id)
    start = timer()
    # mem = memory_usage((write_csv, ('check_out.csv', st))) #, {'sort_order': sort_order.by_id}))
    mem = memory_usage((write_sonata, ('check_out.h5', st)))
    run_time = timer() - start
    for r in range(size):
        if rank == r:
            print('rank {} = {} MB, {} seconds'.format(rank, max(mem), run_time))
            sys.stdout.flush()
        comm.Barrier()
    comm.Barrier()


def save_on_disk():
    np.random.seed(1000)
    st = SpikeTrains(adaptor=STCSVMPIBuffer(cache_dir='tmp_spikes') if size > 1 else STCSVBuffer(cache_dir='tmp_spikes'))
    for i in range(rank, N, size):
        st.add_spikes(node_ids=i, population='v1',
                      timestamps=np.random.uniform(0.0, 3000.0, size=np.random.randint(n_spikes_avg-n_spikes_std, n_spikes_avg+n_spikes_std)))

    if rank == 0:
        print('finished On Disk Version, saving spikes...')
        sys.stdout.flush()

    start = timer()
    # mem = memory_usage((write_csv_old, ('check_out_orig.csv', st), {'sort_order': sort_order.by_id}))
    mem = memory_usage((write_sonata_old, ('check_out_orig.h5', st))) # , {'sort_order': sort_order.by_id}))
    # mem = [10]
    st.close()
    run_time = timer() - start
    for r in range(size):
        if rank == r:
            print('rank {} = {} MB, {} seconds'.format(rank, max(mem), run_time))
            sys.stdout.flush()
        comm.Barrier()
    comm.Barrier()

def save_on_diskv2():
    np.random.seed(1000)
    st = SpikeTrains(adaptor=STCSVMPIBufferV2(cache_dir='tmp_spikes') if size > 1 else STCSVBuffer(cache_dir='tmp_spikes'))
    for i in range(rank, N, size):
        st.add_spikes(node_ids=i, population='v1',
                      timestamps=np.random.uniform(0.0, 3000.0, size=np.random.randint(n_spikes_avg-n_spikes_std, n_spikes_avg+n_spikes_std)))

    if rank == 0:
        print('finished On Disk Version2, saving spikes...')

    start = timer()
    # write_csv_old('check_out1.csv', st, sort_order=sort_order.by_id)
    # mem = memory_usage((write_csv, ('check_out_origv2.csv', st), {'sort_order': sort_order.by_id}))
    mem = memory_usage((write_sonata, ('check_out_origv2.h5', st)))# , {'sort_order': sort_order.by_id}))
    run_time = timer() - start
    for r in range(size):
        if rank == r:
            print('rank {} = {} MB, {} seconds'.format(rank, max(mem), run_time))
            sys.stdout.flush()
        comm.Barrier()
    comm.Barrier()


def add_spikes_tst(adaptor):
    np.random.seed(1000)
    st = SpikeTrains(adaptor)
    for i in range(rank, N, size):
        st.add_spikes(node_ids=i, population='v1',
                      timestamps=np.random.uniform(0.0, 3000.0, size=np.random.randint(n_spikes_avg-n_spikes_std, n_spikes_avg+n_spikes_std)))
    comm.Barrier()
    return st


def add_spike_tst(adaptor):
    np.random.seed(1000)
    st = SpikeTrains(adaptor)
    for node_id in range(rank, N, size):
        for ts in  np.random.uniform(0.0, 3000.0, size=np.random.randint(n_spikes_avg-n_spikes_std, n_spikes_avg+n_spikes_std)):
            st.add_spike(node_id=node_id, population='v1', timestamp=ts)
    comm.Barrier()
    return st


def add_spikes_mem():
    np.random.seed(1000)
    start = timer()
    st = SpikeTrains(adaptor=STMPIBuffer() if size > 1 else STMemoryBuffer())
    for i in range(rank, N, size):
        st.add_spikes(node_ids=i, population='v1',
                      timestamps=np.random.uniform(0.0, 3000.0, size=np.random.randint(n_spikes_avg-n_spikes_std, n_spikes_avg+n_spikes_std)))
    comm.Barrier()
    end = timer()

    n_spikes = st.n_spikes('v1')
    if rank == 0:
        print('In Memory took {} seconds to add {} spikes'.format(end - start, n_spikes))
    comm.Barrier()
    del st

    adaptor = STMPIBuffer() if size > 1 else STMemoryBuffer()
    # mem = memory_usage((add_spikes_tst, (adaptor, )))
    mem = memory_usage((add_spike_tst, (adaptor,)))
    for r in range(size):
        if rank == r:
            print('rank {} = {} MB'.format(rank, max(mem)))
            sys.stdout.flush()
        comm.Barrier()
    comm.Barrier()


def add_spikes_diskv2():
    np.random.seed(1000)
    start = timer()
    st = SpikeTrains(adaptor=STCSVMPIBufferV2(cache_dir='tmp_spikes') if size > 1 else STCSVBuffer(cache_dir='tmp_spikes'))
    for i in range(rank, N, size):
        st.add_spikes(node_ids=i, population='v1',
                      timestamps=np.random.uniform(0.0, 3000.0, size=np.random.randint(n_spikes_avg-n_spikes_std, n_spikes_avg+n_spikes_std)))
    comm.Barrier()
    end = timer()

    if rank == 0:
        #n_spikes = st.n_spikes()
        print('DiskV2 took {} seconds to add spikes'.format(end - start))

try:
    save_in_mem()
    # save_on_disk()
    # save_on_diskv2()
    # single_proc()
    # add_spikes_mem()
    # add_spikes_diskv2()
except Exception as e:
    print(e)
    e.message()


"""
   node_ids population  timestamps
0         0         v1    0.049181
1         0         v1    0.049880
2         0         v1    0.083352
3         0         v1    0.129403
4         0         v1    0.152408
        node_ids population  timestamps
449422      4999         v1    2.879795
449423      4999         v1    2.890416
449424      4999         v1    2.891239
449425      4999         v1    2.957500
449426      4999         v1    2.961819
"""
def single_proc():
    print('--NEW VERSION--')
    psg = PoissonSpikeGenerator(population='v1', seed=10)
    psg.add(node_ids=np.arange(N), firing_rate=30.0, times=3.0)

    tmph5 = tempfile.NamedTemporaryFile(suffix='.h5')
    mem = memory_usage((write_sonata, (tmph5.name, psg)))
    print(max(mem))

    tmph5 = tempfile.NamedTemporaryFile(suffix='.h5')
    start = timer()
    write_sonata(tmph5.name, psg)
    print(timer() - start)

    print('--OLD VERSION--')
    psg = PoissonSpikeGenerator(population='v1', seed=10, buffer_dir='tmp_spikes')
    psg.add(node_ids=np.arange(N), firing_rate=30.0, times=3.0)

    tmph5 = tempfile.NamedTemporaryFile(suffix='.h5')
    mem = memory_usage((write_sonata_old, (tmph5.name, psg)))
    print(max(mem))

    tmph5 = tempfile.NamedTemporaryFile(suffix='.h5')
    start = timer()
    write_sonata_old(tmph5.name, psg)
    print(timer() - start)

