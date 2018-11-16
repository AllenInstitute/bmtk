import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random

#build_recurrent_edges = True

# List of non-virtual cell models
bio_models = [
    {
        'model_name': 'Scnn1a', 'ei': 'e',
        'morphology': 'Scnn1a_473845048_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '472363762_fit.json'
    },
    {
        'model_name': 'Rorb', 'ei': 'e',
        'morphology': 'Rorb_325404214_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863510_fit.json'
    },
    {
        'model_name': 'Nr5a1', 'ei': 'e',
        'morphology': 'Nr5a1_471087815_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863035_fit.json'
    },
    {
        'model_name': 'PV1', 'ei': 'i',
        'morphology': 'Pvalb_470522102_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '472912177_fit.json'
    },
    {
        'model_name': 'PV2', 'ei': 'i',
        'morphology': 'Pvalb_469628681_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473862421_fit.json'
    }
]

point_models = [
    {
        'model_name': 'LIF_exc', 'ei': 'e',
        'dynamics_params': 'IntFire1_exc_1.json'
    },
    {
        'model_name': 'LIF_inh', 'ei': 'i',
        'dynamics_params': 'IntFire1_inh_1.json'
    }
]

# Build a network of 300 biophysical cells to simulate
internal = NetworkBuilder("internal")
for i, model_props in enumerate(bio_models):
    n_cells = 80 if model_props['ei'] == 'e' else 30  # 80% excitatory, 20% inhib

    # Randomly get positions uniformly distributed in a column
    positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)

    internal.add_nodes(N=n_cells,
                       x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                       rotation_angle_yaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  # randomly rotate y axis
                       rotation_angle_zaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  #
                       model_type='biophysical',
                       model_processing='aibs_perisomatic',
                       **model_props)

# Build intfire type cells
for model_props in point_models:
    n_cells = 75  # Just assume 75 cells for both point inhibitory and point excitatory
    positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)
    internal.add_nodes(N=n_cells,
                       x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                       model_type='point_process',
                       model_template='nrn:IntFire1',
                       **model_props)


def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=5):
    """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
    pair will connect the two with a probability prob (excludes self-connections)"""
    if src.node_id == trg.node_id:
        return 0

    return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)


# Connections onto biophysical components, use the connection map to save section and position of every synapse
# exc --> exc connections
internal.add_edges(source={'ei': 'e'}, target={'ei': 'e', 'model_type': 'biophysical'},
                   connection_rule=n_connections,
                   connection_params={'prob': 0.2},
                   dynamics_params='AMPA_ExcToExc.json',
                   model_template='Exp2Syn',
                   syn_weight=6.0e-05,
                   delay=2.0,
                   target_sections=['basal', 'apical'],
                   distance_range=[30.0, 150.0])

# exc --> inh connections
internal.add_edges(source={'ei': 'e'}, target={'ei': 'i', 'model_type': 'biophysical'},
                   connection_rule=n_connections,
                   dynamics_params='AMPA_ExcToInh.json',
                   model_template='Exp2Syn',
                   syn_weight=0.0006,
                   delay=2.0,
                   target_sections=['somatic', 'basal'],
                   distance_range=[0.0, 1.0e+20])

# inh --> exc connections
internal.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'model_type': 'biophysical'},
                   connection_rule=n_connections,
                   dynamics_params='GABA_InhToExc.json',
                   model_template='Exp2Syn',
                   syn_weight=0.002,
                   delay=2.0,
                   target_sections=['somatic', 'basal', 'apical'],
                   distance_range=[0.0, 50.0])

# inh --> inh connections
internal.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'model_type': 'biophysical'},
                   connection_rule=n_connections,
                   connection_params={'prob': 0.2},
                   dynamics_params='GABA_InhToInh.json',
                   model_template='Exp2Syn',
                   syn_weight=0.00015,
                   delay=2.0,
                   target_sections=['somatic', 'basal'],
                   distance_range=[0.0, 1.0e+20])

# For connections on point neurons it doesn't make sense to save syanpatic location
internal.add_edges(source={'ei': 'e'}, target={'model_type': 'point_process'},
                   connection_rule=n_connections,
                   dynamics_params='instanteneousExc.json',
                   syn_weight=0.0019,
                   delay=2.0)

internal.add_edges(source={'ei': 'i'}, target={'model_type': 'point_process'},
                   connection_rule=n_connections,
                   dynamics_params='instanteneousInh.json',
                   syn_weight=0.0019,
                   delay=2.0)

# Build and save internal network
internal.build()
print('Saving internal network')
internal.save(output_dir='network')


# Build a network of 100 virtual cells that will connect to and drive the simulation of the internal network
print('Building external connections')
external = NetworkBuilder("external")

external.add_nodes(N=100, model_type='virtual', ei='e')

# Targets all biophysical excitatory cells
external.add_edges(target=internal.nodes(ei='e', model_type='biophysical'), source=external.nodes(),
                   connection_rule=lambda *_: np.random.randint(0, 5),
                   dynamics_params='AMPA_ExcToExc.json',
                   model_template='Exp2Syn',
                   delay=2.0,
                   syn_weight=0.00041,
                   target_sections=['basal', 'apical', 'somatic'],
                   distance_range=[0.0, 50.0])

# Targets all biophysical inhibitory cells
external.add_edges(target=internal.nodes(ei='i', model_type='biophysical'), source=external.nodes(),
                   connection_rule=lambda *_: np.random.randint(0, 5),
                   dynamics_params='AMPA_ExcToInh.json',
                   model_template='Exp2Syn',
                   delay=2.0,
                   syn_weight=0.00095,
                   target_sections=['basal', 'apical'],
                   distance_range=[0.0, 1e+20])

# Targets all intfire1 cells (exc and inh)
external.add_edges(target=internal.nodes(model_type='point_process'), source=external.nodes(),
                   connection_rule=lambda *_: np.random.randint(0, 5),
                   dynamics_params='instanteneousExc.json',
                   delay=2.0,
                   syn_weight=0.045)


external.build()
print('Saving external')
external.save(output_dir='network')


