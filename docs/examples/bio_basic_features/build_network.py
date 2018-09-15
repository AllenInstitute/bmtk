import os
import numpy as np
from bmtk.builder import NetworkBuilder
from bmtk.utils.io.spike_trains import PoissonSpikesGenerator


build_virtual_net = True

cell_models = [
    {
        'model_name': 'Scnn1a', 'ei': 'e', 'morphology': 'Scnn1a_473845048_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '472363762_fit.json',
    },
    {
        'model_name': 'Rorb', 'ei': 'e', 'morphology': 'Rorb_325404214_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863510_fit.json',
    },
    {
        'model_name': 'Nr5a1', 'ei': 'e', 'morphology': 'Nr5a1_471087815_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863035_fit.json',
    },
    {
        'model_name': 'PV1', 'ei': 'i', 'morphology': 'Pvalb_470522102_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '472912177_fit.json',
    },
    {
        'model_name': 'PV2', 'ei': 'i', 'morphology': 'Pvalb_469628681_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473862421_fit.json',
    }
]

bio_net = NetworkBuilder("bio")

radius = 100.0
dx = 2*np.pi/float(len(cell_models))
for i, model_props in enumerate(cell_models):
    positions = [(radius*np.cos(i*dx), radius*np.sin(i*dx), 0.0)]  # place cells in wheel around origin

    bio_net.add_nodes(model_type='biophysical', model_processing='aibs_perisomatic', positions=positions,
                      **model_props)

bio_net.build()
bio_net.save_nodes(output_dir='network')


if build_virtual_net:
    # Build a separate network of virtual cells to synapse onto the biophysical network
    virt_net = NetworkBuilder('virt')
    virt_net.add_nodes(N=10, model_type='virtual', ei='e')  # 10 excitatory virtual cells
    virt_net.add_edges(target=bio_net.nodes(),  # Connect every virt cells onto every bio cell
                       connection_rule=lambda *_: np.random.randint(4, 12),  # 4 to 12 synapses per source/target
                       dynamics_params='AMPA_ExcToExc.json',
                       model_template='Exp2Syn',
                       syn_weight=3.4e-4,
                       delay=2.0,
                       target_sections=['soma', 'basal', 'apical'],  # target soma and all dendritic sections
                       distance_range=[0.0, 1.0e20])

    virt_net.build()
    virt_net.save(output_dir='network')

    # Create spike trains to use for our virtual cells
    if not os.path.exists('inputs'):
        os.mkdir('inputs')
    psg = PoissonSpikesGenerator(range(10), 10.0, tstop=4000.0)
    psg.to_hdf5('inputs/exc_spike_trains.h5')
