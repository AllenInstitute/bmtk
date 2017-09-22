import os
import sys
from optparse import OptionParser
import h5py
import pandas as pd
import numpy as np

def build_V1():
    # V1 nodes
    node_gids = [0, 1]
    node_type_ids = [0, 1]
    node_groups = [0, 0]
    node_group_indexes = [-1, -1]
    with h5py.File('network/v1_nodes.h5', 'w') as hf:
        hf.create_dataset('nodes/node_type_id', data=node_type_ids, dtype='uint64')
        hf.create_dataset('nodes/node_gid', data=node_gids, dtype='uint64')
        hf.create_dataset('nodes/node_group', data=node_groups, dtype='uint16')
        hf.create_dataset('nodes/node_group_index', data=node_group_indexes, dtype='uint64')
        hf['nodes'].attrs['network'] = 'V1'

        hf.create_group('nodes/0/dynamics_params')

    # V1 recurrent edges
    target_gids = [0, 1]
    source_gids = [1, 0]
    edge_type_ids = [0, 1]
    edge_classs = [0, 0]
    edge_class_index = [-1, -1]
    index_ptr = [0, 1, 2]
    with h5py.File('network/v1_v1_edges.h5', 'w') as hf:
        hf.create_dataset('edges/edge_group', data=edge_classs)

        hf.create_dataset('edges/edge_group_index', data=edge_class_index, dtype='int32')
        hf.create_dataset('edges/edge_type_id', data=edge_type_ids, dtype='int32')
        hf.create_dataset('edges/target_gid', data=target_gids, dtype='uint64')
        hf.create_dataset('edges/source_gid', data=source_gids, dtype='uint64')

        hf['edges/target_gid'].attrs['network'] = 'V1'
        hf['edges/source_gid'].attrs['network'] = 'V1'

        hf.create_dataset('edges/index_pointer', data=index_ptr)

        hf.create_group('edges/0/dynamics_params')

def build_lgn():
    # LGN nodes
    node_gids = [0, 1, 2]
    node_type_ids = [0, 1, 2]
    node_groups = [0, 0, 0]
    node_group_indexes = [-1, -1, -1]
    with h5py.File('network/lgn_nodes.h5', 'w') as hf:
        hf.create_dataset('nodes/node_type_id', data=node_type_ids, dtype='uint64')
        hf.create_dataset('nodes/node_gid', data=node_gids, dtype='uint64')
        hf.create_dataset('nodes/node_group', data=node_groups, dtype='uint16')
        hf.create_dataset('nodes/node_group_index', data=node_group_indexes, dtype='uint64')
        hf['nodes'].attrs['network'] = 'LGN'

        hf.create_group('nodes/0/dynamics_params')

    # LGN --> V1 recurrent edges
    target_gids = [0, 0, 0, 1, 1, 1]
    source_gids = [0, 1, 2, 0, 1, 2]
    edge_type_ids = [0, 0, 0, 1, 1, 1]
    edge_classs = [0, 0, 0, 0, 0, 0]
    edge_class_index = [-1, -1, -1, -1, -1, -1]
    index_pointer = [0, 3, 6]
    with h5py.File('network/lgn_v1_edges.h5', 'w') as hf:
        hf.create_dataset('edges/edge_group', data=edge_classs)

        hf.create_dataset('edges/edge_group_index', data=edge_class_index, dtype='int32')
        hf.create_dataset('edges/edge_type_id', data=edge_type_ids, dtype='int32')
        hf.create_dataset('edges/target_gid', data=target_gids, dtype='uint64')
        hf.create_dataset('edges/source_gid', data=source_gids, dtype='uint64')

        hf['edges/target_gid'].attrs['network'] = 'V1'
        hf['edges/source_gid'].attrs['network'] = 'LGN'

        hf.create_dataset('edges/index_pointer', data=index_pointer)

        hf.create_group('edges/0/dynamics_params')


build_V1()
build_lgn()