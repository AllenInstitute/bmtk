try:
    from bmtk.simulator import pointnet
    import nest

    nest_installed = True
    nest.set_verbosity('M_QUIET')
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.001, "print_time": True})


except ImportError:
    nest_installed = False


class MockNodePop(object):
    def __init__(self, name, nnodes=100, batched=True):
        self.name = name
        self.mixed_nodes = False
        self.internal_nodes_only = True
        self.virtual_nodes_only = False
        self.nnodes = nnodes
        self.batched = batched

    def initialize(self, net):
        pass

    def get_nodes(self):
        if self.batched:
            return [self.MockNode(nnodes=self.nnodes)]
        else:
            return [self.MockNode(nnodes=1) for _ in range(self.nnodes)]

    def filter(self, filter):
        # return [self.MockNode(nnodes=self.nnodes) for _ in range(self.nnodes)]
        return [self.MockNode(nnodes=1, node_id=i) for i in range(self.nnodes)]

    class MockNode(object):
        def __init__(self, nnodes=1, node_id=0):
            self.nnodes = nnodes
            self.node_ids = list(range(nnodes))
            self.node_id = node_id
            self.nest_ids = None

        def build(self):
            self.nest_ids = nest.Create('iaf_psc_delta', self.nnodes, {})


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
            self.nest_params = {'model': 'static_synapse', 'delay': delay,  'weight': 2.0}
