class CellTypes:
    """Essentially an enum to store the type/group of each cell. It's faster and more robust than doing multiple string
    comparisons.
    """
    Point = 0  # any valid nest cell model
    Virtual = 1  # for external nodes
    Other = 2  # should never really get here

    @staticmethod
    def len():
        return 3


class PropertySchema(object):
    #######################################
    # For nodes/cells properties
    #######################################
    def get_cell_type(self, node_params):
        model_type = node_params['model_type'].lower()  # TODO: the column may also be named "level-of-detail"
        if model_type == 'virtual' or model_type == 'filter':
            # external cells
            return CellTypes.Virtual
        else:
            # cell type may be any valid NEST cell model. TODO: validate cell-model with NEST.
            return CellTypes.Point

    def get_params_column(self):
        raise NotImplementedError()

    def get_model_type_column(self):
        return 'model_type'

    #######################################
    # For edge/synapse properties
    #######################################
    def nsyns(self, edge):
        raise NotImplementedError()

    def get_edge_weight(self, src_node, trg_node, edge_props):
        raise NotImplementedError()

