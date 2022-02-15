from bmtk.utils import sonata
from bmtk.utils.reports.spike_trains import SpikeTrains


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
    copy_spikes(pop='lgn')
    copy_spikes(pop='tw')