import numpy as np
from bmtk.builder.networks import NetworkBuilder

net = NetworkBuilder('net')
net.add_nodes(
    N=2,
    x=[0.0]*2,
    y=np.linspace(0.0, -100.0, num=2),
    z=[0.0]*2,
    rotation_angle_xaxis=0.0,
    rotation_angle_yaxis=0.0,
    rotation_angle_zaxis=-3.646878266,
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='472363762_fit.json',
    morphology='Scnn1a_473845048_m.swc'
)

net.add_edges(
    source={'node_id': 0}, target={'node_id': 1},
    connection_rule=20,
    dynamics_params='AMPA_ExcToExc.json',
    model_template='Exp2Syn',
    syn_weight=6.0e-01,
    delay=30.0,
    target_sections=['basal', 'apical', 'soma'],
    distance_range=[0.0, 50.0]
)

net.build()
net.save(output_dir='network_voltage_clamp')
