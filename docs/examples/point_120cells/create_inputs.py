from bmtk.utils.reports.spike_trains import SpikeTrains, PoissonSpikeGenerator

# A constant firing rate of 10 Hz from 0 to 3 seconds
times = (0.0, 3.0)
firing_rate = 10.0

## Uncomment to model the input firing rates on a sin wave
# times = np.linspace(0.0, 3.0, 1000)
# firing_rate = 10.0*np.sin(times) + 10.0

psg = PoissonSpikeGenerator()  # Uses 'seed' to ensure same results every time
psg.add(node_ids='network/thalamus_nodes.h5', firing_rate=firing_rate, times=times, population='thalamus')
psg.to_sonata('thalamus.h5')
psg.to_csv('thalamus.csv')

'''
import numpy as np
import csv

st = SpikeTrains()
with open('thalamus_spikes.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=' ')
    next(csv_reader)
    for r in csv_reader:
        node_id = int(r[0])
        timestamps = np.fromstring(r[1], dtype=np.float64, sep=',')
        st.add_spikes(node_ids=node_id, timestamps=timestamps, population='thalamus')

st.to_sonata('thalamus_spikes.h5')
st.to_csv('thalamus_spikes.old.csv')
'''