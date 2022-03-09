import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.utils.io.spike_trains import PoissonSpikesGenerator
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random
from bmtk.analyzer import nodes_table

# Brunel network parameters -------------------------------------------------------------------------
N_exc = 10000                   # number of excitatory LIF neurons 80%
N_inh = 2500                    # number of inhibitory LIF neurons 20%
n_cells = N_exc + N_inh         # total number of internal neurons
N_ext = 100                    # number of neurons in the external network

JE = 0.1                        # excitatory synaptic strength (mV)
g = 5.                          # ratio of inhibitory synpatic strength to excitatory synaptic strength (unitless)*************
JI = g*JE                       # inhibitory synpatic strength (mv)
eps = 0.1                       # percentage of all possible target neurons that are connected to any given source
CE = int(eps*N_exc)             # number of internal connections with excitatory source neurons 
C_ext = CE                      # number of connections to each internal cell from the input network
n_syn = int(C_ext/N_ext)        # number of synapses per neuron in the input network
CI = int(eps*N_inh)             # number of internal connections with inhibitory source neurons
C = CE + CI                     # number of internal connections
tau_e = 20.                     # membrane time constant of excitatory neurons (ms)
theta = 20.                     # neuron firing threshold (mV)
Vr = 10.                        # neuron rest potential (mV)
tau_rp = 2.                     # refractory period (ms)
D = 1.5                         # transmission delay (ms)
v_ext_v_thr_ratio = 100.        # ratio of v_ext to v_thr (unitless)***************     
eps_ext = C_ext/float(n_cells)  # external connection probability
v_thr = theta/(JE*CE*tau_e)     # frequency needed for a neuron to reach threshold in absence of feedback
v_ext = v_ext_v_thr_ratio*v_thr # external firing frequency 

#-----------------------------------------------------------------------------------------------

def generate_random_positions(N):
    '''
    Generate N random positions.
    N: number of positions to generate
    '''

    x = np.random.random(N)     # x-axis location
    y = np.random.random(N)     # y-axis location
    z = np.random.random(N)     # z-axis location

    positions = np.column_stack((x, y, z))

    return positions

build_recurrent_edges = True

bio_models = {
    "Internal_exc": {   
	'N': N_exc,
        'model_type' :'biophysical',
        'model_name': 'Scnn1a', 'ei': 'e',
        'morphology': 'Scnn1a_473845048_m.swc',
        'model_template': 'nml:Cell_472363762.cell.nml'
    },

    "Internal_inh": {   
	'N': N_inh,
        'model_type' :'biophysical',
        'model_name': 'PV1', 'ei': 'i',
        'morphology': 'Pvalb_470522102_m.swc',
        'model_template': 'nml:Cell_472912177.cell.nml'
    }
}


internal = NetworkBuilder("internal")

for model in bio_models:
    params = bio_models[model].copy()
    internal.add_nodes(**params)

#internal.save_nodes(nodes_file_name='internal_nodes.h5', node_types_file_name='internal_node_types.csv',output_dir='network')    

def random_connections(source,target, p = 0.1 ):
       
    sid = source['node_id']    # Get source id
    tid = target['node_id']    # Get target id
    
    # Avoid self-connections.
    if (sid == tid):
        if sid % 1000 == 0:
            print(sid)
        return None
    return np.random.binomial(1,p) #nsyns

if build_recurrent_edges:
    # exc --> exc connections
    internal.add_edges(source={'ei': 'e'}, target={'ei': 'e'},
                       connection_rule=random_connections,
                       connection_params={'p': eps},
                       dynamics_params='brunel_excitatory.json',
                       model_template='Exp2Syn',
                       syn_weight=7.0e-5,
                       delay=D,
                       target_sections=['basal','apical'],
                       distance_range=[0.0, 1e+20])

    # exc --> inh connections
    internal.add_edges(source={'ei': 'e'}, target={'ei': 'i'},
                       connection_rule=random_connections,
                       connection_params={'p': eps},
                       dynamics_params='brunel_excitatory.json',
                       model_template='Exp2Syn',
                       syn_weight=7.0e-5,
                       delay=D,
                       target_sections=['somatic','basal'],
                       distance_range=[0.0, 1e+20])
    # inh --> exc connections
    internal.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'model_type': 'biophysical'},
                       connection_rule=random_connections,
                       connection_params={'p': eps},
                       dynamics_params='brunel_inhibitory.json',
                       model_template='Exp2Syn',
                       syn_weight=17.5e-5,
                       delay=D,
                       target_sections=['somatic','basal','apical'],
                       distance_range=[0.0, 200])

    # inh --> inh connections
    internal.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'model_type': 'biophysical'},
                       connection_rule=random_connections,
                       connection_params={'p': eps},
                       dynamics_params='brunel_inhibitory.json',
                       model_template='Exp2Syn',
                       syn_weight=17.5e-5,
                       delay=D,
                       target_sections=['somatic','basal'],
                       distance_range=[0.0, 1e+20])

internal.build()

print('Saving internal')
internal.save(output_dir='network')
print('Building external connections')
external = NetworkBuilder("external")
external.add_nodes(N=N_ext, model_type='virtual', ei='e')

cm = external.add_edges(source=external.nodes(),
                        target=internal.nodes(),
                        connection_rule=random_connections,
                        connection_params={'p': 0.1},
                        dynamics_params='AMPA_ExcToExc.json',
                        model_template='Exp2Syn',
                        syn_weight = 7.0e-3,
                        delay=D,
                        target_sections = ['somatic','basal'],
                        distance_range=[0.0, 1e+20])

external.build()
print('Saving external')
external.save(output_dir='network')




