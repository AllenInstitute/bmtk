import pytest
import numpy as np


from bmtk.utils.reports.spike_trains import SpikeTrains
from bmtk.utils.reports.spike_trains import sort_order


def test_load_sonata():
    st = SpikeTrains(population_name='net1', use_caching=True)
    st.add_spikes(spike_times=np.linspace(0, 1000.0, 30), node_id=0)
    st.add_spikes(spike_times=[0.3, 0.6, 0.9], node_id=1)
    st.add_spike(spike_time=0.5, node_id=3)
    st.add_spike(spike_time=1.0, node_id=3)
    st.close()

    print(st.population == 'net1')
    print(st.n_ids)
    print(st.n_spikes)
    print(st.node_ids)
    print(list(st.spikes()))
    #st.to_csv('here.csv', sort_order=sort_order.by_time)


def test_load_csv():
    st = SpikeTrains.from_csv('spikes.csv', population='test')
    print(st.node_ids)
    print(st.n_ids)
    print(st.get_spike_times(node_id=0))
    #print(st.get_spike_times(node_id=0, time_window=(400.0, 700.0)))
    #print(st.get_spikes_df(population='NONE', node_ids=[0, 1, 2], sort_by=sort_order.by_id, time_window=(200.0, 400.0)))
    print(st.get_spikes_df(node_ids=[0, 1, 2], sort_by=sort_order.by_id))




if __name__ == '__main__':
    #test_load_sonata()
    test_load_csv()