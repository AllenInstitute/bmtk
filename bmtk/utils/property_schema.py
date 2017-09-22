"""


"""

# TODO: go through the individual simulator's property_schemas and pull out the common functionality. Ideally all
#       simulators should share ~80% of the same schema, with some differences in how certain columns are determined.
# TODO: Add access to builder so when a network is built with a given property schema
# TODO: have utils.io.tabular_network use these schemas to discover name of node-id, node-type-id, etc for different
#       standards.
class PropertySchema:
    pass