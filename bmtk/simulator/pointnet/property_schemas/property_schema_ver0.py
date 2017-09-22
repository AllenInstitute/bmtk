from base_schema import CellTypes, PropertySchema as BaseSchema


class PropertySchema(BaseSchema):
    def get_cell_type(self, node_params):
        model_type = node_params['model_type'].lower()  # TODO: the column may also be named "level-of-detail"
        if model_type == 'virtual' or model_type == 'filter' or model_type == 'spike_generator':
            # external cells
            return CellTypes.Virtual
        else:
            # cell type may be any valid NEST cell model. TODO: validate cell-model with NEST.
            return CellTypes.Point

    def get_params_column(self):
        return 'params_file'

    def nsyns(self, edge):
        if 'nsyns' in edge:
            return edge['nsyns']
        return 1

    def get_edge_weight(self, src_node, trg_node, edge_props):
        weight_fnc = edge_props['weight_max']
        # TODO: reimplement with global modules like nrn
        weight_fnc = edge_props._graph.get_weight_function(edge_props['weight_function'])
        # TODO: should always be target, source - not the other way around (will cause problems)
        return weight_fnc(trg_node, src_node, edge_props)
