from base_schema import PropertySchema as BaseSchema


class PropertySchema(BaseSchema):
    def get_params_column(self):
        return 'dynamics_params'

    #######################################
    # For edge/synapse properties
    #######################################
    def nsyns(self, edge):
        return 1

    def get_edge_weight(self, src_node, trg_node, edge_props):
        return edge_props['syn_weight']
