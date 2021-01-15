from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

psg = PoissonSpikeGenerator(population='LGN')
psg.add(
    node_ids=range(0, 100),
    firing_rate=8.0,    # we can also pass in a nonhomoegenous function/array
    times=(0.0, 2.0)    # Firing starts at 0 s up to 3 s
)

psg.to_sonata('inputs/lgn_spikes.poisson.h5')
psg.to_csv('inputs/lgn_spikes.poisson.csv')