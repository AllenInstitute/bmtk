import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.bionet.swc_reader import get_swc
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator


cell_models = [
    {
        'model_name': 'Scnn1a',
        'ei': 'e',
        'morphology': 'Scnn1a_473845048_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '472363762_fit.json',
    },
    {
        'model_name': 'Rorb',
        'ei': 'e',
        'morphology': 'Rorb_325404214_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863510_fit.json',
    },
    {
        'model_name': 'Nr5a1',
        'ei': 'e',
        'morphology': 'Nr5a1_471087815_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863035_fit.json',
    },
    {
        'model_name': 'PV1',
        'ei': 'i',
        'morphology': 'Pvalb_470522102_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '472912177_fit.json',
    },
    {
        'model_name': 'PV2',
        'ei': 'i',
        'morphology': 'Pvalb_469628681_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473862421_fit.json',
    }
]

bio_net = NetworkBuilder("bio")

radius = 100.0
dx = 2*np.pi/float(len(cell_models))
for i, model_props in enumerate(cell_models):
    bio_net.add_nodes(
        model_type='biophysical',
        model_processing='aibs_perisomatic',
        x=[radius*np.cos(i*dx)],
        y=[radius*np.sin(i*dx)],
        z=[0.0],
        **model_props
    )

bio_net.build()
bio_net.save_nodes(output_dir='network')


# Build a separate network of virtual cells to synapse onto the biophysical network
def set_synapses(src, trg, section_names=('soma', 'apical', 'basal'), distance_range=(0.0, 1.0e20)):
    trg_swc = get_swc(trg, morphology_dir='../bio_components/morphologies/', use_cache=True)
    sec_ids, seg_xs = trg_swc.choose_sections(section_names, distance_range, n_sections=1)
    return [sec_ids[0], seg_xs[0]]


virt_net = NetworkBuilder('virt')
virt_net.add_nodes(N=10, model_type='virtual', ei='e')  # 10 excitatory virtual cells
cm = virt_net.add_edges(
    target=bio_net.nodes(),  # Connect every virt cells onto every bio cell
    connection_rule=lambda *_: np.random.randint(4, 12),  # 4 to 12 synapses per source/target
    dynamics_params='AMPA_ExcToExc.json',
    model_template='Exp2Syn',
    syn_weight=3.4e-4,
    delay=2.0,
)

cm.add_properties(['afferent_section_id', 'afferent_section_pos'], rule=set_synapses, dtypes=[np.int, np.float])

virt_net.build()
virt_net.save(output_dir='network')


#
activator_net = NetworkBuilder("activator")
activator_net.add_nodes(
    N=1,
    x=[0.0],
    y=[0.0],
    z=[0.0],
    model_name='Scnn1a',
    ei='e',
    morphology='Scnn1a_473845048_m.swc',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='472363762_fit.json',
    model_type='biophysical',
    model_processing='aibs_perisomatic',
)
cm = activator_net.add_edges(
    target=bio_net.nodes(ei='e'),  # Connect every virt cells onto every bio cell
    connection_rule=lambda *_: np.random.randint(4, 12),  # 4 to 12 synapses per source/target
    dynamics_params='AMPA_ExcToExc.json',
    model_template='Exp2Syn',
    syn_weight=3.4e-4,
    delay=2.0,
)
cm.add_properties(['afferent_section_id', 'afferent_section_pos'], rule=set_synapses, dtypes=[np.int, np.float])

cm = activator_net.add_edges(
    target=bio_net.nodes(ei='e'),
    connection_rule=lambda *_: np.random.randint(4, 12),
    dynamics_params='AMPA_ExcToInh.json',
    model_template='Exp2Syn',
    syn_weight=3.4e-4,
    delay=2.0,
)
cm.add_properties(['afferent_section_id', 'afferent_section_pos'], rule=set_synapses, dtypes=[np.int, np.float])

activator_net.build()
activator_net.save(output_dir='network')


if not os.path.exists('inputs/virt_spikes.h5'):
    psg = PoissonSpikeGenerator()
    psg.add(node_ids=[n['node_id'] for n in bio_net.nodes()], firing_rate=10.0, times=(1.0, 4.0), population='virt')
    psg.to_sonata('inputs/virt_spikes.h5')
