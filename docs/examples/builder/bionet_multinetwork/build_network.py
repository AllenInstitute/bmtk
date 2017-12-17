from bmtk.builder.networks import NetworkBuilder

def crule(src, trg):
    # print src.node_id, trg.node_id
    return 2

net1 = NetworkBuilder('NET1')
net1.add_nodes(N=100, position=[(0.0, 1.0, -1.0)] * 100, cell_type='Scnna1', ei='e')
net1.add_edges(source={'ei': 'e'}, target={'ei': 'e'}, connection_rule=5)
net1.build()

net2 = NetworkBuilder('NET2')
net2.add_nodes(N=10, position=[(0.0, 1.0, -1.0)] * 10, cell_type='PV1', ei='i')
net2.add_edges(connection_rule=10)
net2.add_edges(source=net1.nodes(), connection_rule=1)
net2.add_edges(target=net1.nodes(), connection_rule=crule)
net2.build()

#net1.save_edges(output_dir='tmp_output')
net2.save_edges(output_dir='tmp_output')