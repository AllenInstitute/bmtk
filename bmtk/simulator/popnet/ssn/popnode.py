

class SSNNode:
    def __init__(self, population, node_id, gid, **node_properties):
        self.population = population
        self.node_id = node_id 
        self.gid = gid

        self.type = None
        self.input_offset = []
        self.scaling_coef = [] 
        self.exponent = []
        self.decay_const = []
        self.init_value = 0.0
        self.external_inputs = None
        self.node_properties = node_properties
        self._sonata_node = self.node_properties.get('node', {})

    def __contains__(self, property):
        return property in self.node_properties or property in self._sonata_node

    def __getitem__(self, property):
        if property in self.node_properties:
            return self.node_properties[property]
        elif property in self._sonata_node:
            return self._sonata_node[property]
        elif property in ['node_id', 'node_ids']:
            return self.node_id
        elif property in ['population', 'population_name']:
            return self.population
        else:
            raise KeyError(f'{self.__class__.__name__} does not contain property "{property}"')

    def get(self, property, default=None):
        if property in self:
            return self[property]
        else:
            return default

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {self.gid} > ({self.population}.{self.node_id})'