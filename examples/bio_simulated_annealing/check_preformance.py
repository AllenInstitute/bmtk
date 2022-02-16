from bmtk.simulator import bionet
from neuron import h

from memory_profiler import memory_usage


pc = h.ParallelContext()


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()


mem = memory_usage((run, ('config.simulation.json',)))
print(max(mem))
pc.done()
