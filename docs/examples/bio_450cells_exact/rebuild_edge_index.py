import h5py
import numpy as np

with h5py.File('network/lgn_v1_edges.h5', 'a') as h5:
    node_id_data = np.zeros((449, 2), dtype=np.uint64)
    ds = h5['edges/lgn_to_v1/indices/target_to_source/node_id_to_range']
    node_id_data[0:449, :] = ds
    node_id_data[448,:] = [448, 448]

    #print node_id_data
    #ds.resize((449, 2))
    #ds[...] = node_id_data
    #ds.resize((449, 2))
    del h5['edges/lgn_to_v1/indices/target_to_source/node_id_to_range']

    h5.create_dataset('edges/lgn_to_v1/indices/target_to_source/node_id_to_range', data=node_id_data)
    #ary = np.array(h5['edges/lgn_to_v1/indices/target_to_source/node_id_to_range'][...])
    #print np.append(ary, [[448, 449]])