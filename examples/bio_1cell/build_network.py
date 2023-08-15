import os
import numpy as np

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.bionet.swc_reader import get_swc, SWCReader
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

build_virtual_cells = True
np.random.seed(100)


net = NetworkBuilder('biocell')
net.add_nodes( 
    pop_name='Scnn1a',
    x=[0.0],
    y=[-100.0],
    z=[0.0],
    rotation_angle_xaxis=[0.0],
    rotation_angle_yaxis=[0.0],
    rotation_angle_zaxis=[3.646878266],
    tuning_angle=np.random.uniform(0.0, 360.0, size=1),
    model_type='biophysical',
    model_template='nml:Cell_472363762.cell.nml',
    model_processing='aibs_perisomatic',
    morphology='Scnn1a_473845048_m.swc'
)
net.build()
net.save(output_dir='network')


if build_virtual_cells:
    virt = NetworkBuilder('virt')
    virt.add_nodes(
        model_type='virtual'
    )
    cm = virt.add_edges(
        source=virt.nodes(), target=net.nodes(),
        connection_rule=12,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='Exp2Syn',
        syn_weight=0.035,
        delay=2.0
    )

    swc = SWCReader('components/morphologies/Scnn1a_473845048_m.swc')
    swc.fix_axon()
    sec_ids, seg_xs = swc.choose_sections(['soma', 'dend', 'apic'], [0.0, 10.0e20], n_sections=12)
    cm.add_properties(
        'afferent_section_id',
        values=sec_ids,
        dtypes=int
    )
    cm.add_properties(
        'afferent_section_pos',
        values=seg_xs,
        dtypes=float
    )
    virt.build()
    virt.save(output_dir='network')

    if not os.path.exists('inputs/virt_spikes.h5'):
        psg = PoissonSpikeGenerator()
        psg.add(node_ids='network/virt_nodes.h5', firing_rate=20.0, times=(0.0, 3.0), population='virt')
        psg.to_sonata('inputs/virt_spikes.h5')
        psg.to_csv('inputs/virt_spikes.csv')
