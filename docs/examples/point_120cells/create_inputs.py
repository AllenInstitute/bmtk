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
