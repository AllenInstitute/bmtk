try:
    from bmtk.simulator import pointnet
    from bmtk.simulator.pointnet.nest_utils import NEST_SYNAPSE_MODEL_PROP
    import nest

    nest_installed = True
    nest.set_verbosity('M_QUIET')
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.001, "print_time": True})


except ImportError:
    nest_installed = False


class MockNodePop(object):
    def __init__(self, name, nnodes=100, batched=True, virtual=False):
        self.name = name
        self.mixed_nodes = False
        self.internal_nodes_only = not virtual
        self.virtual_nodes_only = virtual
        self.n_nodes = nnodes
        self.batched = batched

    def initialize(self, net):
        pass

    def get_nodes(self):
        if self.batched:
            return [self.MockNode(n_nodes=self.n_nodes)]
        else:
            return [self.MockNode(n_nodes=1) for _ in range(self.n_nodes)]

    def filter(self, filter):
        # return [self.MockNode(nnodes=self.nnodes) for _ in range(self.nnodes)]
        return [self.MockNode(n_nodes=1, node_id=i) for i in range(self.n_nodes)]

    class MockNode(object):
        def __init__(self, n_nodes=1, node_id=0):
            self.n_nodes = n_nodes
            self.node_ids = list(range(n_nodes))
            self.node_id = node_id
            self.nest_ids = None

        def build(self):
            self.nest_ids = nest.Create('iaf_psc_delta', self.n_nodes, {})


class MockEdges(object):
    def __init__(self, name, source_nodes, target_nodes, delay=2.0):
        self.name = name
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self._connection_type = 0
        self.virtual_connections = False
        self.delay = delay

    def initialize(self, net):
        pass

    def set_connection_type(self, src_pop, trg_pop):
        pass

    def get_edges(self):
        return [self.MockEdge(delay=self.delay)]

    class MockEdge(object):
        def __init__(self, delay):
            self.source_node_ids = range(100)
            self.target_node_ids = range(100)
            self.nest_params = {NEST_SYNAPSE_MODEL_PROP: 'static_synapse', 'delay': [delay]*100,  'weight': [2.0]*100}


class MockNodeSet(object):
    def __init__(self, population_names):
        self._population_names = population_names

    def population_names(self):
        return self._population_names


class MockSpikes(object):
    def __init__(self, spike_times):
        self.spikes = spike_times

    def get_times(self, node_id):
        return self.spikes
