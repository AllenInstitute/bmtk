import numpy as np

from bmtk.utils.reports.spike_trains import SpikeTrains, PoissonSpikeGenerator


def create_inputs_const(firing_rate_hz, times=(0.0, 3.0)):
    psg = PoissonSpikeGenerator()
    psg.add(node_ids='network/external_nodes.h5', firing_rate=firing_rate_hz, times=times, population='external')
    psg.to_sonata('inputs/external_spikes.h5')
    # psg.to_csv('inputs/external_spikes.csv')


def create_inputs_sigmoid(min_rate=1.0, max_rate=10.0, onset=0.5, ts_begin=0.0, ts_end=3.0):
    times = np.linspace(ts_begin, ts_end, 1000)
    rates = (max_rate-min_rate)/(1.0 + np.exp(-(times-onset)*20)) + min_rate

    psg = PoissonSpikeGenerator()
    psg.add(node_ids='network/external_nodes.h5', firing_rate=rates, times=times, population='external')
    psg.to_sonata('inputs/external_spikes.h5')
    # psg.to_csv('inputs/external_spikes.csv')


def create_inputs_lgn_spikes():
    lgn_spikes = SpikeTrains.load('../spikes_inputs/lgn_spikes.h5')

    spikes_writer = SpikeTrains()
    for i in range(0, 100):
        spikes_writer.add_spikes(node_ids=i, timestamps=lgn_spikes.get_times(i), population='external')

    spikes_writer.to_sonata('inputs/external_spikes.h5')
    # spikes_writer.to_csv('inputs/external_spikes.csv')


if __name__ == '__main__':
    # create_inputs_const(firing_rate_hz=10.0)
    create_inputs_sigmoid()
    # create_inputs_lgn_spikes()
