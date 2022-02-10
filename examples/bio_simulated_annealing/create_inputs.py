import numpy as np

from bmtk.utils import sonata
from bmtk.utils.reports.spike_trains import SpikeTrains, PoissonSpikeGenerator


def create_inputs_const(pop, firing_rate_hz, times=(0.0, 3.0)):
    psg = PoissonSpikeGenerator()
    psg.add(node_ids='network/{}_nodes.h5'.format(pop), firing_rate=firing_rate_hz, times=times, population=pop)
    psg.to_sonata('inputs/{}_spikes.h5'.format(pop))
    # psg.to_csv('inputs/{}_spikes.csv'.format(pop)


def create_inputs_sigmoid(pop, min_rate=1.0, max_rate=10.0, onset=0.5, ts_begin=0.0, ts_end=3.0):
    times = np.linspace(ts_begin, ts_end, 1000)
    rates = (max_rate-min_rate)/(1.0 + np.exp(-(times-onset)*20)) + min_rate

    psg = PoissonSpikeGenerator()
    psg.add(node_ids='network/{}_nodes.h5'.format(pop), firing_rate=rates, times=times, population='external')
    psg.to_sonata('inputs/{}_spikes.h5'.format(pop))
    # psg.to_csv('inputs/{}_spikes.csv'.format(pop)


def copy_spikes(pop):
    net = sonata.File(
        data_files='network/{}_nodes.h5'.format(pop),
        data_type_files='network/{}_node_types.csv'.format(pop)
    )
    node_ids = net.nodes[pop].node_ids

    orig_spikes = SpikeTrains.load('../spikes_inputs/{}_spikes.h5'.format(pop))
    spikes_writer = SpikeTrains()
    for node_id in node_ids:
        spikes_writer.add_spikes(node_ids=node_id, timestamps=orig_spikes.get_times(node_id), population=pop)

    spikes_writer.to_sonata('inputs/{}_spikes.h5'.format(pop))
    # spikes_writer.to_csv('inputs/{}_spikes.csv'.format(pop))


if __name__ == '__main__':
    # create_inputs_const(firing_rate_hz=10.0)
    # create_inputs_sigmoid()
    copy_spikes(pop='tw')
