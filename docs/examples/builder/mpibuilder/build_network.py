import os
import sys
from optparse import OptionParser
import numpy as np
import pandas as pd
import h5py

from bmtk.builder.networks import MPIBuilder


net = MPIBuilder("V1/L4")
net.add_nodes(N=2, pop_name='Scnn1a', node_type_id=395830185,
              positions=[(28.753, -364.868, -161.705), (48.753, -344.868, -141.705)],
              tuning_angle=[0.0, 25.0],
              rotation_angle_yaxis=[3.55501, 3.55501],
              location='VisL4',
              ei='e',
              level_of_detail='biophysical',
              params_file='472363762_fit.json',
              morphology_file='Scnn1a-Tg3-Cre_Ai14_IVSCC_-177300.01.02.01_473845048_m.swc',
              rotation_angle_zaxis=-3.646878266,
              set_params_function='Biophys1')

net.add_nodes(N=2, pop_name='Rorb', node_type_id=314804042,
              positions=[(241.092, -349.263, 146.916), (201.092, -399.263, 126.916)],
              tuning_angle=[50.0, 75.0],
              rotation_angle_yaxis=[3.50934, 3.50934],
              location='VisL4',
              ei='e',
              level_of_detail='biophysical',
              params_file='473863510_fit.json',
              morphology_file='Rorb-IRES2-Cre-D_Ai14_IVSCC_-168053.05.01.01_325404214_m.swc',
              rotation_angle_zaxis=-4.159763785,
              set_params_function='Biophys1')

net.add_nodes(N=2, pop_name='Nr5a1', node_type_id=318808427,
              positions=[(320.498, -351.259, 20.273), (310.498, -371.259, 10.273)],
              tuning_angle=[100.0, 125.0],
              rotation_angle_yaxis=[0.72202, 0.72202],
              location='VisL4',
              ei='e',
              level_of_detail='biophysical',
              params_file='473863035_fit.json',
              morphology_file='Nr5a1-Cre_Ai14_IVSCC_-169250.03.02.01_471087815_m.swc',
              rotation_angle_zaxis=-2.639275277,
              set_params_function='Biophys1')

net.add_nodes(N=2, pop_name='PV1', node_type_id=330080937,
              positions=[(122.373, -352.417, -216.748), (102.373, -342.417, -206.748)],
              tuning_angle=['NA', 'NA'],
              rotation_angle_yaxis=[2.92043, 2.92043],
              location='VisL4',
              ei='i',
              level_of_detail='biophysical',
              params_file='472912177_fit.json',
              morphology_file='Pvalb-IRES-Cre_Ai14_IVSCC_-176847.04.02.01_470522102_m.swc',
              rotation_angle_zaxis=-2.539551891,
              set_params_function='Biophys1')

net.add_nodes(N=2, pop_name='PV2', node_type_id=318331342,
              positions=[(350.321, -372.535, -18.282), (360.321, -371.535, -12.282)],
              tuning_angle=['NA', 'NA'],
              rotation_angle_yaxis=[5.043336, 5.043336],
              location='VisL4',
              ei='i',
              level_of_detail='biophysical',
              params_file='473862421_fit.json',
              morphology_file='Pvalb-IRES-Cre_Ai14_IVSCC_-169125.03.01.01_469628681_m.swc',
              rotation_angle_zaxis=-3.684439949,
              set_params_function='Biophys1')

net.add_nodes(N=2, pop_name='LIF_exc', node_type_id=100000101,
              positions=[(-243.04, -342.352, -665.666), (-233.04, -332.352, -675.666)],
              tuning_angle=['NA', 'NA'],
              #rotation_angle_yaxis=[5.11801, 5.11801],
              location='VisL4',
              ei='e',
              level_of_detail='intfire',
              params_file='IntFire1_exc_1.json',
              set_params_function='IntFire1')

net.add_nodes(N=2, pop_name='LIF_inh', node_type_id=100000102,
              positions=[(211.04, -321.333, -631.593), (218.04, -327.333, -635.593)],
              tuning_angle=[150.0, 175.0],
              #rotation_angle_yaxis=[4.566091, 4.566091],
              location='VisL4',
              ei='i',
              level_of_detail='intfire',
              params_file='IntFire1_inh_1.json',
              set_params_function='IntFire1')

net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'level_of_detail': 'biophysical'},
              connection_rule=5,
              weight_max=0.0002,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              params_file='GABA_InhToInh.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'level_of_detail': 'intfire'},
              connection_rule=5,
              weight_max=0.00225,
              weight_function='wmax',
              delay=2.0,
              params_file='instanteneousInh.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'level_of_detail': 'biophysical'},
              connection_rule=lambda trg, src: 5,
              weight_max=0.00018,
              weight_function='wmax',
              distance_range=[0.0, 50.0],
              target_sections=['somatic', 'basal', 'apical'],
              delay=2.0,
              params_file='GABA_InhToExc.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'level_of_detail': 'intfire'},
              connection_rule=5,
              weight_max=0.009,
              weight_function='wmax',
              delay=2.0,
              params_file='instanteneousInh.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV1'},
              connection_rule=5,
              weight_max=0.00035,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              params_file='AMPA_ExcToInh.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV2'},
              connection_rule=5,
              weight_max=0.00027,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              params_file='AMPA_ExcToInh.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_inh'},
              connection_rule=5,
              weight_max=0.0043,
              weight_function='wmax',
              delay=2.0,
              params_file='instanteneousExc.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Scnn1a'},
              connection_rule=5,
              weight_max=6.4e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              params_file='AMPA_ExcToExc.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Rorb'},
              connection_rule=5,
              weight_max=5.5e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              params_file='AMPA_ExcToExc.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Nr5a1'},
              connection_rule=5,
              weight_max=7.2e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              params_file='AMPA_ExcToExc.json',
              set_params_function='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
              connection_rule=5,
              weight_max=0.0019,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              delay=2.0,
              params_file='instanteneousExc.json',
              set_params_function='exp2syn')

cm = net.add_edges(source={'ei': 'i'}, target={'ei': 'e'}, connection_rule=lambda s, t: 5, p1='e2i', p2='e2i')
cm.add_properties(names=['segment', 'distance'], rule=lambda s, t: [1, 0.5], dtypes=[np.int, np.float])

net.build()

# print net.edges(source_nodes=0, weight_function='gaussianLLL')
net.save_nodes(nodes_file_name='output/v1_nodes.h5', node_types_file_name='output/v1_node_types.csv')
# print net.edges()
net.save_edges(edges_file_name='output/v1_v1_edges.h5', edge_types_file_name='output/v1_v1_edge_types.csv')
