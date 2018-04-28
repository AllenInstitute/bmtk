from bmtk.utils import sonata

sf = sonata.File(data_files='network/v1_nodes.h5', data_type_files='network/v1_node_types.csv')
sf.nodes.generate_gids('network/gids.h5')