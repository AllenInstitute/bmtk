import pytest
import numpy as np

from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator


def test_psg_fixed():
    psg = PoissonSpikeGenerator(population='test', seed=100)
    psg.add(node_ids=range(10), firing_rate=5.0, times=(0.0, 3.0))
    assert(psg.populations == ['test'])
    assert(np.all(psg.node_ids() == list(range(10))))
    assert(psg.n_spikes() == 143)
    assert(psg.n_spikes(population='test') == 143)
    assert(np.allclose(psg.time_range(), (0.005380662350673328, 2.9865205688893295)))

    df = psg.to_dataframe()
    assert(df.shape == (143, 3))

    assert(np.allclose(psg.get_times(node_id=0), [0.156, 0.222, 0.332, 0.705, 0.706, 0.731, 0.954, 1.303, 1.333, 1.504,
                                                  1.948, 1.995, 2.036, 2.059, 2.108, 2.877], atol=1.0e-3))
    assert(np.allclose(psg.get_times(node_id=9, population='test'),
                       [0.0233, 0.241, 0.390, 0.428, 1.001, 1.056, 2.424, 2.599, 2.640, 2.737, 2.780, 2.885],
                       atol=1.0e-3))


def test_psg_variable():
    times = np.linspace(0.0, 3.0, 1000)
    fr = np.exp(-np.power(times - 1.0, 2) / (2*np.power(.5, 2)))*5

    psg = PoissonSpikeGenerator(population='test', seed=0.0)
    psg.add(node_ids=range(10), firing_rate=fr, times=times)

    assert(psg.populations == ['test'])
    assert(np.all(psg.node_ids() == list(range(10))))
    assert(psg.n_spikes() == 59)
    assert(np.allclose(psg.time_range(), (0.13932107933711294, 2.9013003727909172)))
    assert(psg.to_dataframe().shape == (59, 3))
    assert(np.allclose(psg.get_times(node_id=0), [0.442, 0.520, 0.640, 1.099, 1.393, 1.725], atol=1.0e-3))
    assert(np.allclose(psg.get_times(node_id=9), [0.729, 0.885, 1.047, 1.276, 1.543, 1.669, 1.881], atol=1.0e-3))


if __name__ == '__main__':
    test_psg_fixed()
    test_psg_variable()
