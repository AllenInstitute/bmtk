import itertools
import pandas as pd
import model_data

from bmtk.builder import NetworkBuilder


params = pd.read_csv("demo_params.csv", index_col=0, header=None).squeeze("columns")


## Create L23 network with recurrent connections
l23_net = NetworkBuilder('l23')
l23_net.add_nodes(
    pop_name='Exc',
    model_template='ssn:Recurrent',
    model_type='rate_population',
    scaling_coef=params['conn_scale'],
    input_offset=0.0,
    exponent=2.0,
    decay_const=params['tau_e']*1000.0,
    # dynamics_params='exc_ssn.json'
)
l23_net.add_nodes(
    pop_name='PV',
    model_template='ssn:Recurrent',
    model_type='rate_population',
    scaling_coef=params['conn_scale'],
    input_offset=0.0,
    exponent=2.0,
    decay_const=params['tau_p']*1000.0,
    # dynamics_params='pv_ssn.json'
)

l23_net.add_nodes(
    pop_name='SST',
    model_template='ssn:Recurrent',
    model_type='rate_population',
    scaling_coef=params['conn_scale'],
    input_offset=0.0,
    exponent=2.0,
    decay_const=params['tau_s']*1000.0,
    # dynamics_params='sst_ssn.json'
)

l23_net.add_nodes(
    pop_name='VIP',
    model_template='ssn:Recurrent',
    model_type='rate_population',
    scaling_coef=params['conn_scale'],
    input_offset=0.0,
    exponent=2.0,
    decay_const=params['tau_v']*1000.0,
    # dynamics_params='vip_ssn.json'
)


for src, trg in itertools.product(['Exc', 'PV', 'SST', 'VIP'], repeat=2):
    lu_key = f'{src[0]}_to_{trg[0]}'.lower()
    if lu_key in params:
        l23_net.add_edges(
            source={'pop_name': src}, target={'pop_name': trg},
            syn_weight=params[lu_key]*model_data.l23_infl_df.loc[lu_key]['mean']
        )
        # print(src, trg, params[lu_key]*model_data.l23_infl_df.loc[lu_key]['mean'])


l23_net.build()
l23_net.save(output_dir='network')


## Create L4 -> L23 Connections
l4e_net = NetworkBuilder('l4e')
l4e_net.add_nodes(
    model_template='ssn:External',
    model_type='virtual'
)

for trg, syn_weight in zip(['Exc', 'PV', 'SST', 'VIP'], params[['stim_e', 'stim_p', 'stim_s', 'stim_v']].values*model_data.e4_l23_infl_matrix_mean):
    l4e_net.add_edges(
        target=l23_net.nodes(pop_name=trg),
        syn_weight=syn_weight
    )


l4e_net.build()
l4e_net.save(output_dir='network')


## Create BKG -> L23 Connections
bkg_net = NetworkBuilder('bkg')
bkg_net.add_nodes(
    model_template='ssn:External',
    model_type='virtual'
)
for trg, syn_weight in zip(['Exc', 'PV', 'SST', 'VIP'], params[['input_e', 'input_p', 'input_s', 'input_v']].values / params["conn_scale"]):
    bkg_net.add_edges(
        target=l23_net.nodes(pop_name=trg),
        syn_weight=syn_weight
    )
bkg_net.build()
bkg_net.save(output_dir='network')
