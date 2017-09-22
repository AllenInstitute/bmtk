import bmtk.simulator.utils.config as config
from bmtk.simulator.pointnet.graph import Graph
from bmtk.simulator.pointnet.network import Network
import weight_funcs as fn

configure = config.from_json('config.json')
graph = Graph(configure)
graph.add_weight_function(fn.wmax)
graph.add_weight_function(fn.gaussianLL)

net = Network(configure, graph)
net.run(configure['run']['duration'])



