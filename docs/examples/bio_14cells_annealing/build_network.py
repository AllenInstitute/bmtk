import os
import numpy as np

from bmtk.builder.networks import NetworkBuilder


# Step 1: Create a v1 mock network of 14 cells (nodes) with across 7 different cell "types"
net = NetworkBuilder("v1")
net.add_nodes(N=2,  # specifiy the number of cells belong to said group.
              pop_name='Scnn1a', location='VisL4', ei='e',  # pop_name, location, and ei are optional parameters that help's identifies properties of the cells. The modeler can choose whatever key-value pairs as they deem appropiate.
              positions=[(28.753, -364.868, -161.705),  # The following properties we are passing in lists
                         (48.753, -344.868, -141.705)], # of size N. Doing so will uniquely assign different
              tuning_angle=[0.0, 25.0],                 #  values to each individual cell
              rotation_angle_yaxis=[3.55501, 3.81531],
              rotation_angle_zaxis=-3.646878266, # Note that the y-axis rotation is differnt for each cell (ie. given a list of size N), but with z-axis rotation all cells have the same value
              model_type='biophysical',  # The type of cell we are using
              model_template='ctdb:Biophys1.hoc',  # Tells the simulator that when building cells models use a hoc_template specially created for parsing Allen Cell-types file models. Value would be different if we were using NeuronML or different model files
              model_processing='aibs_perisomatic',  # further instructions for how to processes a cell model. In this case aibs_perisomatic is a built-in directive to cut the axon in a specific way
              dynamics_params='472363762_fit.json',  # Name of file (downloaded from Allen Cell-Types) used to set model parameters and channels
              morphology='Scnn1a-Tg3-Cre_Ai14_IVSCC_-177300.01.02.01_473845048_m.swc')  # Name of morphology file downloaded

net.add_nodes(N=2, pop_name='Rorb', location='VisL4', ei='e',
              positions=[(241.092, -349.263, 146.916), (201.092, -399.263, 126.916)],
              tuning_angle=[50.0, 75.0],
              rotation_angle_yaxis=[3.50934, 3.50934],
              model_type='biophysical',
              model_template='ctdb:Biophys1.hoc',
              model_processing='aibs_perisomatic',
              dynamics_params='473863510_fit.json',
              morphology='Rorb-IRES2-Cre-D_Ai14_IVSCC_-168053.05.01.01_325404214_m.swc',
              rotation_angle_zaxis=-4.159763785)

net.add_nodes(N=2, pop_name='Nr5a1', location='VisL4', ei='e',
              positions=[(320.498, -351.259, 20.273), (310.498, -371.259, 10.273)],
              tuning_angle=[100.0, 125.0],
              rotation_angle_yaxis=[0.72202, 0.72202],
              model_type='biophysical',
              model_template='ctdb:Biophys1.hoc',
              model_processing='aibs_perisomatic',
              dynamics_params='473863035_fit.json',
              morphology='Nr5a1-Cre_Ai14_IVSCC_-169250.03.02.01_471087815_m.swc',
              rotation_angle_zaxis=-2.639275277)

# Note that in the previous cells we set the tuning_angle, but for PV1 and PV2 such parameter is absent (as it is not
# applicable for inhibitory cells). The BMTK builder allows heterogeneous cell properties as dictated by the model
net.add_nodes(N=2, pop_name='PV1', location='VisL4', ei='i',
              positions=[(122.373, -352.417, -216.748), (102.373, -342.417, -206.748)],
              rotation_angle_yaxis=[2.92043, 2.92043],
              model_type='biophysical',
              model_template='ctdb:Biophys1.hoc',
              model_processing='aibs_perisomatic',
              dynamics_params='472912177_fit.json',
              morphology='Pvalb-IRES-Cre_Ai14_IVSCC_-176847.04.02.01_470522102_m.swc',
              rotation_angle_zaxis=-2.539551891)

net.add_nodes(N=2, pop_name='PV2', location='VisL4', ei='i',
              positions=[(350.321, -372.535, -18.282), (360.321, -371.535, -12.282)],
              rotation_angle_yaxis=[5.043336, 5.043336],
              model_type='biophysical',
              model_template='ctdb:Biophys1.hoc',
              model_processing='aibs_perisomatic',
              dynamics_params='473862421_fit.json',
              morphology='Pvalb-IRES-Cre_Ai14_IVSCC_-169125.03.01.01_469628681_m.swc',
              rotation_angle_zaxis=-3.684439949)


# Along with our biophysical cells our network will also include integate-and-fire point cells
net.add_nodes(N=2, pop_name='LIF_exc', location='VisL4', ei='e',
              positions=[(-243.04, -342.352, -665.666), (-233.04, -332.352, -675.666)],
              tuning_angle=[150.0, 175.0],
              model_type='point_process',  # use point_process to indicate were are using point model cells
              model_template='nrn:IntFire1',  # Tell the simulator to use the NEURON built-in IntFire1 type cell
              dynamics_params='IntFire1_exc_1.json')

net.add_nodes(N=2, pop_name='LIF_inh', location='VisL4', ei='i',
              positions=[(211.04, -321.333, -631.593), (218.04, -327.333, -635.593)],
              model_type='point_process',
              model_template='nrn:IntFire1',
              dynamics_params='IntFire1_inh_1.json')


# Step 2: We want to connect our network. Just like how we have node-types concept we group our connections into
# "edge-types" that share rules and properties
net.add_edges(source={'ei': 'i'},  # For our synaptic source cells we select all inhibitory cells (ei==i), incl. both biophys and point
              target={'ei': 'i', 'model_type': 'biophysical'},  # For our synaptic target we select all inhibitory biophysically detailed cells
              connection_rule=5,  # All matching source/target pairs will have
              syn_weight=0.0002,  # synaptic weight
              target_sections=['somatic', 'basal'],  # Gives the simulator the target sections and
              distance_range=[0.0, 1e+20],           # distances (from soma) when creating connections
              delay=2.0,
              dynamics_params='GABA_InhToInh.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'model_type': 'point_process'},
              connection_rule=5,
              syn_weight=0.00225,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instanteneousInh.json')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'model_type': 'biophysical'},
              connection_rule=lambda trg, src: 5,
              syn_weight=0.00018,
              weight_function='wmax',
              distance_range=[0.0, 50.0],
              target_sections=['somatic', 'basal', 'apical'],
              delay=2.0,
              dynamics_params='GABA_InhToExc.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'model_type': 'point_process'},
              connection_rule=5,
              syn_weight=0.009,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instanteneousInh.json')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV1'},
              connection_rule=5,
              syn_weight=0.00035,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              dynamics_params='AMPA_ExcToInh.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV2'},
              connection_rule=5,
              syn_weight=0.00027,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              dynamics_params='AMPA_ExcToInh.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_inh'},
              connection_rule=5,
              syn_weight=0.0043,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instanteneousExc.json')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Scnn1a'},
              connection_rule=5,
              syn_weight=6.4e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Rorb'},
              connection_rule=5,
              syn_weight=5.5e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Nr5a1'},
              connection_rule=5,
              syn_weight=7.2e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
              connection_rule=5,
              syn_weight=0.0019,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              delay=2.0,
              dynamics_params='instanteneousExc.json')


net.build()
net.save(output_dir='network')


def generate_positions(N, x0=0.0, x1=300.0, y0=0.0, y1=100.0):
    X = np.random.uniform(x0, x1, N)
    Y = np.random.uniform(y0, y1, N)
    return np.column_stack((X, Y))


def select_source_cells(src_cells, trg_cell, n_syns):
    if 'tuning_angle' in trg_cell:
        synapses = [n_syns if src['pop_name'] == 'tON' or src['pop_name'] == 'tOFF' else 0 for src in src_cells]
    else:
        synapses = [n_syns if src['pop_name'] == 'tONOFF' else 0 for src in src_cells]

    return synapses


lgn = NetworkBuilder("lgn")
lgn.add_nodes(N=30, pop_name='tON', ei='e', location='LGN',
              positions=generate_positions(30),
              model_type='virtual')

lgn.add_nodes(N=30, pop_name='tOFF', ei='e', location='LGN',
              positions=generate_positions(30),
              model_type='virtual')

lgn.add_nodes(N=30, pop_name='tONOFF', ei='e', location='LGN',
              positions=generate_positions(30),
              model_type='virtual')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='Rorb'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'n_syns': 10},
              syn_weight=5e-05,
              weight_function='wmax',
              distance_range=[0.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='Nr5a1'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'n_syns': 10},
              syn_weight=5e-05,
              weight_function='wmax',
              distance_range=[0.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='Scnn1a'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'n_syns': 10},
              syn_weight=4e-05,
              weight_function='wmax',
              distance_range=[0.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='PV1'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'n_syns': 10},
              syn_weight=0.0001,
              weight_function='wmax',
              distance_range=[0.0, 1.0e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              dynamics_params='AMPA_ExcToInh.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='PV2'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'n_syns': 10},
              syn_weight=9e-05,
              weight_function='wmax',
              distance_range=[0.0, 1.0e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              dynamics_params='AMPA_ExcToInh.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='LIF_exc'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'n_syns': 10},
              syn_weight=0.0045,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instanteneousExc.json')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='LIF_inh'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'n_syns': 10},
              syn_weight=0.002,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instanteneousExc.json')

lgn.build()
lgn.save(output_dir='network')


tw = NetworkBuilder("tw")
tw.add_nodes(N=30, pop_name='TW', ei='e', location='TW', model_type='virtual')

tw.add_edges(source=tw.nodes(), target=net.nodes(pop_name='Rorb'),
             connection_rule=5,
             syn_weight=0.00015,
             weight_function='wmax',
             distance_range=[30.0, 150.0],
             target_sections=['basal', 'apical'],
             delay=2.0,
             dynamics_params='AMPA_ExcToExc.json',
             model_template='exp2syn')

tw.add_edges(source=tw.nodes(), target=net.nodes(pop_name='Scnn1a'),
             connection_rule=5,
             syn_weight=0.00019,
             weight_function='wmax',
             distance_range=[30.0, 150.0],
             target_sections=['basal', 'apical'],
             delay=2.0,
             dynamics_params='AMPA_ExcToExc.json',
             model_template='exp2syn')

tw.add_edges(source=tw.nodes(), target=net.nodes(pop_name='Nr5a1'),
             connection_rule=5,
             syn_weight=0.00019,
             weight_function='wmax',
             distance_range=[30.0, 150.0],
             target_sections=['basal', 'apical'],
             delay=2.0,
             dynamics_params='AMPA_ExcToExc.json',
             model_template='exp2syn')

tw.add_edges(source=tw.nodes(), target=net.nodes(pop_name='PV1'),
             connection_rule=5,
             syn_weight=0.0022,
             weight_function='wmax',
             distance_range=[0.0, 1.0e+20],
             target_sections=['basal', 'somatic'],
             delay=2.0,
             dynamics_params='AMPA_ExcToInh.json',
             model_template='exp2syn')

tw.add_edges(source=tw.nodes(), target=net.nodes(pop_name='PV2'),
             connection_rule = 5,
             syn_weight = 0.0013,
             weight_function = 'wmax',
             distance_range = [0.0, 1.0e+20],
             target_sections = ['basal', 'somatic'],
             delay = 2.0,
             dynamics_params = 'AMPA_ExcToInh.json',
             model_template = 'exp2syn')

tw.add_edges(source=tw.nodes(), target=net.nodes(pop_name='LIF_exc'),
             connection_rule=5,
             syn_weight=0.015,
             weight_function='wmax',
             delay=2.0,
             dynamics_params='instanteneousExc.json')

tw.add_edges(source=tw.nodes(), target=net.nodes(pop_name='LIF_inh'),
             connection_rule=5,
             syn_weight=0.05,
             weight_function='wmax',
             delay=2.0,
             dynamics_params='instanteneousExc.json')

tw.build()
tw.save(output_dir='network')
