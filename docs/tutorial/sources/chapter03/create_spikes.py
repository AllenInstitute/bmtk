from bmtk.utils.spike_trains import SpikesGenerator

sg = SpikesGenerator(nodes='network/mthalamus_nodes.h5', t_max=3.0)
sg.set_rate(10.0)
sg.save_csv('thalamus_spikes.csv', in_ms=True)