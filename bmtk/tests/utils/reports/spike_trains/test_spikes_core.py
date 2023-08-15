import pytest
import numpy as np

from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
from bmtk.utils.reports.spike_trains import SpikeTrains


def test_psg_fixed():
    psg = PoissonSpikeGenerator(population='test', seed=100)
    psg.add(node_ids=range(10), firing_rate=5.0, times=(0.0, 3.0))
    assert(psg.populations == ['test'])
    assert(np.all(psg.node_ids() == list(range(10))))
    assert(psg.n_spikes() == 143)
    assert(psg.n_spikes(population='test') == 143)
    assert (np.allclose(psg.time_range(), (5.380662350673328, 2986.5205688893295)))

    df = psg.to_dataframe()
    assert(df.shape == (143, 3))

    assert(np.allclose(psg.get_times(node_id=0), [156.7916, 222.0400, 332.5493, 705.1267, 706.0727, 731.9963, 954.1834,
                                                  1303.7542, 1333.1543, 1504.3314, 1948.2045, 1995.1471, 2036.1411,
                                                  2059.0835, 2108.6982, 2877.7935], atol=1.0e-3))

    assert(np.allclose(psg.get_times(node_id=9, population='test'),
                       [23.3176, 241.7332, 390.1951, 428.2215, 1001.0229, 1056.4003, 2424.8442, 2599.6312, 2640.1228,
                        2737.9504, 2780.0140, 2885.8020], atol=1.0e-3))


def test_psg_variable():
    times = np.linspace(0.0, 3.0, 1000)
    fr = np.exp(-np.power(times - 1.0, 2) / (2*np.power(.5, 2)))*5

    psg = PoissonSpikeGenerator(population='test', seed=0.0)
    psg.add(node_ids=range(10), firing_rate=fr, times=times)

    assert(psg.populations == ['test'])
    assert(np.all(psg.node_ids() == list(range(10))))
    assert(psg.n_spikes() == 59)
    assert(np.allclose(psg.time_range(), (139.32107933711294, 2901.3003727909172)))
    assert(psg.to_dataframe().shape == (59, 3))
    assert(np.allclose(psg.get_times(node_id=0), [442.8378, 520.3624, 640.3880, 1099.0661, 1393.0794, 1725.6109],
                       atol=1.0e-3))
    assert(np.allclose(psg.get_times(node_id=9), [729.6267, 885.2469, 1047.7728, 1276.3554, 1543.6557, 1669.9070,
                                                  1881.3605], atol=1.0e-3))


def test_equals():
    st1 = SpikeTrains()
    st1.add_spikes(node_ids=0, population='V1', timestamps=[0.1, 0.2, 0.3, 0.4])
    st1.add_spikes(node_ids=1, population='V1', timestamps=[1.0])

    st2 = SpikeTrains()
    st2.add_spikes(node_ids=1, population='V1', timestamps=[1.0])
    st2.add_spikes(node_ids=0, population='V1', timestamps=[0.3, 0.2, 0.1, 0.4])

    assert(st1 == st2)
    assert(st1 <= st2)
    assert(st1 >= st2)
    assert(not st1 != st2)


def test_subset():
    st1 = SpikeTrains()
    st1.add_spikes(node_ids=0, population='V1', timestamps=[0.1, 0.2, 0.3, 0.4])
    st1.add_spikes(node_ids=1, population='V1', timestamps=[1.0])

    st2 = SpikeTrains()
    st2.add_spikes(node_ids=1, population='V1', timestamps=[1.0])
    st2.add_spikes(node_ids=0, population='V1', timestamps=[0.3, 0.2, 0.1, 0.4, 0.5])
    st2.add_spike(node_id=2, population='V1', timestamp=0.5)
    st2.add_spikes(node_ids=0, population='V2', timestamps=np.linspace(0.0, 1.0, 11))

    assert(st1 != st2)
    assert(st1 < st2)
    assert(st2 > st1)
    assert(st1 <= st2)
    assert(st2 >= st1)


if __name__ == '__main__':
    test_psg_fixed()
    # test_psg_variable()
    # test_equals()
    # test_subset()
