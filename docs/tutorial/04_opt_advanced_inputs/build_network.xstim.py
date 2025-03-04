import os
import numpy as np

from bmtk.builder import NetworkBuilder


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
bio_net.save_nodes(output_dir='network_xstim')
