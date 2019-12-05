import pytest
import os
import numpy as np

from bmtk.utils.reports.spike_trains import SpikeTrains, pop_na
from bmtk.utils.reports.spike_trains import spike_train_buffer


def full_path(file_path):
    cpath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cpath, file_path)


@pytest.mark.parametrize('path', ['spike_files/spikes.old.h5'])
def test_old_populations(path):
    path = full_path(path)
    st = SpikeTrains.from_sonata(full_path(path))
    assert(st.populations == [pop_na])
    node0_timestamps = st.get_times(node_id=0, population=pop_na)
    assert(len(node0_timestamps) > 0)
    assert(np.all(st.get_times(node_id=0) == node0_timestamps))
    assert(np.all(st.get_times(node_id=0, population='should_still_work') == node0_timestamps))


@pytest.mark.parametrize('path', ['spike_files/spikes.one_pop.h5'])
def test_single_populations(path):
    path = full_path(path)
    st = SpikeTrains.from_sonata(path)
    assert(st.populations == ['v1'])
    node0_timestamps = st.get_times(node_id=0, population='v1')

    assert(np.all(st.get_times(node_id=0) == node0_timestamps))
    assert(st.get_times(node_id=0, population='should_not_work') == [])


@pytest.mark.parametrize('path', ['spike_files/spikes.multipop.h5'])
def test_multi_populations(path):
    path = full_path(path)
    st = SpikeTrains.from_sonata(path)
    assert('tw' in st.populations and 'lgn' in st.populations)
    n1_tw_ts = st.get_times(node_id=0, population='tw')
    n1_lgn_ts = st.get_times(node_id=0, population='lgn')

    assert(len(n1_tw_ts) > 0)
    assert(len(n1_lgn_ts) > 0)
    assert(not np.array_equal(n1_tw_ts, n1_lgn_ts))  # (np.any(n1_tw_ts != n1_lgn_ts))
    assert(st.get_times(node_id=0, population='other') == [])


@pytest.mark.parametrize('path', ['spike_files/spikes.multipop.h5'])
def test_multipop_with_default(path):
    path = full_path(path)
    st = SpikeTrains.from_sonata(path, population='tw')
    assert('tw' in st.populations and 'lgn' not in st.populations)
    n1_tw_ts = st.get_times(node_id=0, population='tw')
    assert(len(n1_tw_ts) > 0)
    assert(np.all(n1_tw_ts == st.get_times(node_id=0)))


def test_empty_spikes():
    st = SpikeTrains(adaptor=spike_train_buffer.STMemoryBuffer())
    output_path = full_path('output/tmpspikes.h5')
    st.to_sonata(path=output_path)
    st.close()

    st_empty = SpikeTrains.from_sonata(output_path)
    assert(st_empty.populations == [])
    assert(st_empty.n_spikes() == 0)
    assert(list(st_empty.spikes()) == [])
    os.remove(output_path)


if __name__ == '__main__':
    #test_old_populations('spike_files/spikes.old.h5')
    #test_single_populations('spike_files/spikes.one_pop.h5')
    test_multi_populations('spike_files/spikes.multipop.h5')
    #test_multipop_with_default('spike_files/spikes.multipop.h5')
    #test_empty_spikes()