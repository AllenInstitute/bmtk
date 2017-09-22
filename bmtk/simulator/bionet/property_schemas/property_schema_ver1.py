from .. import nrn
from base_schema import CellTypes, PropertySchema as BaseSchema


class PropertySchema(BaseSchema):

    def get_cell_type(self, node_params):
        # TODO: A lookup table may be faster
        model_type = node_params['model_type']
        if model_type == 'biophysical':
            return CellTypes.Biophysical
        elif model_type == 'point_IntFire1':
            return CellTypes.Point
        elif model_type == 'virtual':
            return CellTypes.Virtual
        else:
            return CellTypes.Unknown

    def get_positions(self, node_params):
        if 'positions' in node_params:
            return node_params['positions']
        else:
            return None

    def get_edge_weight(self, src_node, trg_node, edge):
        # TODO: check to see if weight function is None or non-existant
        return edge['syn_weight']
        #weight_fnc = nrn.py_modules.synaptic_weight(edge['syn_weight'])
        #return weight_fnc(trg_node, src_node, edge)

    def preselected_targets(self):
        return True

    def target_sections(self, edge):
        if 'sec_id' in edge:
            return edge['sec_id']
        return None

    def target_distance(self, edge):
        if 'sec_x' in edge:
            return edge['sec_x']
        return None

    def nsyns(self, edge):
        if 'nsyns' in edge:
            return edge['nsyns']
        return 1

    def get_params_column(self):
        return 'dynamics_params'

    def model_type(self, node):
        return node['model_type']

    """
    @staticmethod
    def load_cell_hobj(node):
        print PropertySchema.model_type(node)
        exit()
        #model_type_str = PropertySchema.model_type_str(node)
        #print model_type_str
        # print node.model_params
        # print model_type_str
        cell_fnc = nrn.py_modules.cell_model(model_type_str)
        return cell_fnc(node)
        #if model_type_str in nrn.py_modules.cell_models:
        #   nrn.py_modules.cell_model()
        # print node.model_params
        #exit()
    """
