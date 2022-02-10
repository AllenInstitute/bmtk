import pandas as pd
import numpy as np

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.bionet.swc_reader import get_swc


net = NetworkBuilder("v1")
net.add_nodes(
    N=1,
    pop_name='Scnn1a',
    ei='e',
    location='VisL4',
    x=[200.0],
    y=[-368.0],
    z=[0.0],
    rotation_angle_xaxis=[0.0],
    rotation_angle_yaxis=[0.0],
    rotation_angle_zaxis=[3.8291696219999998],
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='476126528_fit.json',
    morphology='Scnn1a-Tg3-Cre_Ai14-187849.05.02.01_491119823_m'
)

net.add_nodes(
    N=1,
    pop_name='Pvalb',
    ei='i',
    location='VisL4',
    x=[-200.0],
    y=[-412.12],
    z=[0.0],
    rotation_angle_xaxis=[0.0],
    rotation_angle_yaxis=[0.0],
    rotation_angle_zaxis=[3.975017803],
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='333604946_fit.json',
    morphology='Pvalb-IRES-Cre_Ai14-176848.03.01.01_491119484_m'
)


def set_synapses(src, trg):
    trg_swc = get_swc(trg, morphology_dir='components/morphologies/', use_cache=True)

    sec_ids, seg_xs = trg_swc.choose_sections(['dend', 'apic'], [0.0, 10.0e20], n_sections=1)
    sec_id, seg_x = sec_ids[0], seg_xs[0]
    swc_id, swc_dist = trg_swc.get_swc_id(sec_id, seg_x)
    # coords = trg_swc.get_coords(sec_id, seg_x)

    return [sec_id, seg_x, swc_id, swc_dist]


cm = net.add_edges(
    source={'ei': 'e'}, target={'ei': 'i'},
    connection_rule=lambda *_: np.random.randint(5, 15),
    dynamics_params='AMPA_ExcToInh.json',
    model_template='Exp2Syn',
    syn_weight=0.035,
    delay=2.0
)
cm.add_properties(
    ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
    rule=set_synapses,
    dtypes=[np.int, np.float, np.int, np.float]
)

cm = net.add_edges(
    source={'ei': 'i'}, target={'ei': 'e'},
    connection_rule=lambda *_: np.random.randint(5, 15),
    dynamics_params='GABA_InhToExc.json',
    model_template='Exp2Syn',
    syn_weight=0.018,
    delay=2.0
)
cm.add_properties(
    ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
    rule=set_synapses,
    dtypes=[np.int, np.float, np.int, np.float]
)

net.build()
net.save(output_dir='network')
