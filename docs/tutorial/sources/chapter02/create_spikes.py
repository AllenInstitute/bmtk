from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

psg = PoissonSpikeGenerator(population='mthalamus')
psg.add(
    node_ids=range(10),  # Have 10 nodes to match mthalamus
    firing_rate=10.0,    # 10 Hz, we can also pass in a nonhomoegenous function/array
    times=(0.0, 3.0)    # Firing starts at 0 s up to 3 s
)
psg.to_sonata('inputs/mthalamus_spikes.h5')
