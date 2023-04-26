import matplotlib.pyplot as plt
import numpy as np

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.bionet.swc_reader import get_swc
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator


def create_spokes_nsyns(src_neurons, trg_neuron):
    # For each point_neuron on the outer ring set n random connections to the central biophysical neuron
    nsyns_vals = [np.random.randint(1, 12) for _ in range(len(src_neurons))]
    return nsyns_vals


def set_synapses(src, trg):
    trg_swc = get_swc(trg, morphology_dir='components/morphologies/', use_cache=True)

    sec_ids, seg_xs = trg_swc.choose_sections(['soma', 'dend', 'apic'], [0.0, 10.0e20], n_sections=1)
    sec_id, seg_x = sec_ids[0], seg_xs[0]
    swc_id, swc_dist = trg_swc.get_swc_id(sec_id, seg_x)
    # coords = trg_swc.get_coords(sec_id, seg_x)

    return [sec_id, seg_x, swc_id, swc_dist]


def build_network(ring_size=6, ring_radius=50):
    net = NetworkBuilder("v1")
    net.add_nodes(
        N=1,
        pop_name='Rbp4',
        ei='e',
        location='VisL5',
        x=[0.0],
        y=[-475.0],
        z=[0.0],
        rotation_angle_xaxis=[0.0],
        rotation_angle_yaxis=[0.0],
        rotation_angle_zaxis=[2.527764895],
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        model_processing='aibs_perisomatic',
        dynamics_params='471129934_fit.json',
        morphology='Rbp4-Cre_KL100_Ai14-180747.06.01.01_495335491_m'
    )

    # Create a uniform ring of intfire neuron encircling the biophys cell (in x/z plane)
    angles_rad = np.linspace(0.0, 2*np.pi, num=ring_size, endpoint=False)
    net.add_nodes(
        N=ring_size,
        pop_name='Rbp4',
        ei='e',
        location='VisL5',
        x=[ring_radius*np.cos(r) for r in angles_rad],
        y=[-475.0]*ring_size,
        z=[ring_radius*np.sin(r) for r in angles_rad],
        model_type='point_neuron',
        model_template='nrn:IntFire1',
        dynamics_params='e5Rbp4_avg_lif.json',
    )

    cm = net.add_edges(
        source={'model_type': 'point_neuron'}, target={'model_type': 'biophysical'},
        connection_rule=create_spokes_nsyns,
        iterator='all_to_one',
        dynamics_params='AMPA_ExcToExc.json',
        model_template='Exp2Syn',
        syn_weight=0.045,
        delay=10.0
    )
    cm.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=set_synapses,
        dtypes=[np.int, np.float, np.int, np.float]
    )

    net.build()
    net.save(output_dir='network')
    return net


def build_virt_cells(net):
    point_neurons = list(net.nodes(model_type='point_neuron'))
    virt_net = NetworkBuilder("virtual")
    virt_net.add_nodes(N=len(point_neurons), model_type='virtual', ei='e')
    virt_net.add_edges(
        source=virt_net.nodes(), target=net.nodes(model_type='point_neuron'),
        connection_rule=lambda src, trg: 1 if src.node_id == trg.node_id-1 else 0,
        dynamics_params='instantaneousExc.json',
        delay=0.0,
        syn_weight=10.0
    )

    virt_net.build()
    virt_net.save(output_dir='network')
    return virt_net


def create_inputs(virt_net, firing_rate=30.0, times=(0.0, 3.0)):
    virt_node_ids = [v['node_id'] for v in virt_net.nodes()]
    psg = PoissonSpikeGenerator()
    psg.add(node_ids=virt_node_ids, firing_rate=firing_rate, times=times, population=virt_net.name)
    psg.to_sonata('inputs/virtual_spikes.h5')


if __name__ == '__main__':
    net = build_network()
    virt_net = build_virt_cells(net)
    create_inputs(virt_net)




# 471129934 biophysical Rbp4-Cre_KL100_Ai14-180747.06.01.01_495335491_m.swc 471129934_fit.json -2.527764895 v1 e5Rbp4 VisL5 e aibs_perisomatic ctdb:Biophys1.hoc