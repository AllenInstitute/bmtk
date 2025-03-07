# import os
import numpy as np
# import pandas as pd

from bmtk.builder import NetworkBuilder
# from bmtk.builder.auxi.node_params import positions_columinar, xiter_random



np.random.seed(100)

visp = NetworkBuilder('VISp')
visp.add_nodes(
    N=80,
    model_name='Pyr',
    ei='e',
    model_type='biophysical',
    model_processing='aibs_perisomatic',
    morphology='Scnn1a_473845048_m.swc',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='472363762_fit.json'
)

visp.add_nodes(
    N=20,
    model_name='Pvalb',
    ei='i',
    model_type='biophysical',
    model_processing='aibs_perisomatic',
    morphology='Pvalb_470522102_m.swc',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='472912177_fit.json'
)

# exc --> exc connections
visp.add_edges(
    source={'ei': 'e'}, target={'ei': 'e'},
    connection_rule=lambda *_: np.random.randint(0, 10),
    dynamics_params='AMPA_ExcToExc.json',
    model_template='Exp2Syn',
    syn_weight=6.0e-05,
    delay=2.0,
    target_sections=['basal', 'apical'],
    distance_range=[30.0, 150.0]
)

# exc --> inh connections
visp.add_edges(
    source={'ei': 'e'}, target={'ei': 'i'},
    connection_rule=lambda *_: np.random.randint(0, 10),
    dynamics_params='AMPA_ExcToInh.json',
    model_template='Exp2Syn',
    syn_weight=0.0006,
    delay=2.0,
    target_sections=['somatic', 'basal'],
    distance_range=[0.0, 1.0e+20]
)

# inh --> exc connections
visp.add_edges(
    source={'ei': 'i'}, target={'ei': 'e'},
    connection_rule=lambda *_: np.random.randint(0, 10),
    dynamics_params='GABA_InhToExc.json',
    model_template='Exp2Syn',
    syn_weight=0.002,
    delay=2.0,
    target_sections=['somatic', 'basal', 'apical'],
    distance_range=[0.0, 50.0]
)

# inh --> inh connections
visp.add_edges(
    source={'ei': 'i'}, target={'ei': 'i'},
    connection_rule=lambda *_: np.random.randint(0, 10),
    dynamics_params='GABA_InhToInh.json',
    model_template='Exp2Syn',
    syn_weight=0.00015,
    delay=2.0,
    target_sections=['somatic', 'basal'],
    distance_range=[0.0, 1.0e+20]
)

visp.build()
visp.save(output_dir='network_spont_syns')
