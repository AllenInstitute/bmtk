# import os
import numpy as np
# import pandas as pd

from bmtk.builder import NetworkBuilder



np.random.seed(100)

cortex = NetworkBuilder('cortex')
cortex.add_nodes(
    N=10,
    model_name='Pyr',
    ei='e',
    model_type='biophysical',
    model_processing='aibs_perisomatic',
    morphology='Scnn1a_473845048_m.swc',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='472363762_fit.json'
)

cortex.build()
cortex.save(output_dir='network_csv_spikes')


### Stimuli into the VISp from LGN
inputs = NetworkBuilder('inputs')
inputs.add_nodes(
    N=10,
    model_type='virtual',
    ei='e'
)
inputs.add_edges(
    target=cortex.nodes(), source=inputs.nodes(),
    connection_rule=lambda s, t: np.random.uniform(10, 30) if s['node_id'] == t['node_id'] else 0,
    dynamics_params='AMPA_ExcToExc.json',
    model_template='Exp2Syn',
    delay=2.0,
    syn_weight=0.0041,
    target_sections=['basal', 'apical', 'somatic'],
    distance_range=[0.0, 10.0e20]
)
inputs.build()
inputs.save(output_dir='network_csv_spikes')






