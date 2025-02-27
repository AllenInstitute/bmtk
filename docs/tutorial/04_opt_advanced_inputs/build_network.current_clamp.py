import numpy as np
from bmtk.builder.networks import NetworkBuilder

net = NetworkBuilder('net')
net.add_nodes(
    N=5,
    x=[0.0]*5,
    y=np.linspace(0.0, -100.0, num=5),
    z=[0.0]*5,
    rotation_angle_xaxis=0.0,
    rotation_angle_yaxis=0.0,
    rotation_angle_zaxis=-3.646878266,
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='472363762_fit.json',
    morphology='Scnn1a_473845048_m.swc'
)
net.build()
net.save(output_dir='network_current_clamp')
