from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

psg = PoissonSpikeGenerator(population='mthalamus')
psg.add(node_ids=range(100),  # Have 100 nodes to match mthalamus
        firing_rate=15.0,    # 15 Hz, we can also pass in a nonhomogeneous function/array
        times=(0.0, 3.0))    # Firing starts at 0 s up to 3 s
psg.to_sonata('inputs/mthalamus_spikes.h5')
