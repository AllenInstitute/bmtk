from bmtk.simulator import bionet
from bmtk.utils.reports.spike_trains import SpikeTrains

config_file = 'config.simulation.json'

conf = bionet.Config.from_json(config_file, validate=True)
# conf.build_env()
graph = bionet.BioNetwork.from_config(conf)
graph.build()
node_props = graph.node_properties()
node_ids = {k: v.tolist() for k, v in node_props['v1'].groupby('pop_name').groups.items()}
print(node_ids)

st = SpikeTrains.load('output/spikes.h5')
