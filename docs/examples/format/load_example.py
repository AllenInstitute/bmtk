import os
from bmtk.utils.io.tabular_network_v1 import NodesFile, EdgesFile

HOME = '/allen/aibs/mat/Kael/'

nf = NodesFile()
nf.load(os.path.join(HOME, 'V1Model_July21/networks/V1/v1_nodes.h5'),
        os.path.join(HOME, 'V1Model_July21/networks/V1/v1_node_types.csv'))

#for row in nf:
#    print row.gid, row.group, row['model_type']



print nf.groups[0].columns


#print len(nf)
#print nf.name
#print nf.version
#print len(nf)
#node = nf.get_node(6)
#print node
#print node.gid
#print node.group

exit()

ef = EdgesFile()
ef.load('/home/kael/Data/Data/V1Model_July21/networks/V1/v1_edges.h5',
        '/home/kael/Data/Data/V1Model_July21/networks/V1/v1_edge_types.csv')

print('{} --> {}'.format(ef.source_network, ef.target_network))
print('{} total edges'.format(len(ef)))
#print(ef.edge_types_table)
#print ef[5]
n_edges = 0
sources = set()
for e in ef.edges_itr(target_gid=1):
    assert(e.target_gid == 1)
    n_edges += 1
    sources.add(e.source_gid)
print('{} total edges from {} sources.'.format(n_edges, len(sources)))
