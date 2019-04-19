import pytest
import numpy as np

from bmtk.utils.reports.spike_trains import SpikeTrains, pop_na


def test_old_populations(path):
    st = SpikeTrains.from_sonata(path)
    assert(st.populations == [pop_na])
    node0_timestamps = st.get_times(node_id=0, population=pop_na)
    assert(len(node0_timestamps) > 0)
    assert(np.all(st.get_times(node_id=0) == node0_timestamps))
    assert(np.all(st.get_times(node_id=0, population='should_still_work') == node0_timestamps))


def test_single_populations(path):
    st = SpikeTrains.from_sonata(path)
    assert(st.populations == ['v1'])
    node0_timestamps = st.get_times(node_id=0, population='v1')

    assert(np.all(st.get_times(node_id=0) == node0_timestamps))
    assert(st.get_times(node_id=0, population='should_not_work') == [])


def test_multi_populations(path):
    st = SpikeTrains.from_sonata(path)
    assert('tw' in st.populations and 'lgn' in st.populations)
    n1_tw_ts = st.get_times(node_id=0, population='tw')
    n1_lgn_ts = st.get_times(node_id=0, population='lgn')

    assert(len(n1_tw_ts) > 0)
    assert(len(n1_lgn_ts) > 0)
    assert(np.any(n1_tw_ts != n1_lgn_ts))
    assert(st.get_times(node_id=0, population='other') == [])


def test_multipop_with_default(path):
    st = SpikeTrains.from_sonata(path, population='tw')
    assert('tw' in st.populations and 'lgn' not in st.populations)
    n1_tw_ts = st.get_times(node_id=0, population='tw')
    assert(len(n1_tw_ts) > 0)
    assert(np.all(n1_tw_ts == st.get_times(node_id=0)))


if __name__ == '__main__':
    test_old_populations('spike_files/spikes.old.h5')
    test_single_populations('spike_files/spikes.one_pop.h5')
    test_multi_populations('spike_files/spikes.multipop.h5')
    test_multipop_with_default('spike_files/spikes.multipop.h5')
