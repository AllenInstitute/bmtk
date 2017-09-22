import json
import ast
import numpy as np

from .. import nrn
from base_schema import CellTypes, PropertySchema as BaseSchema


class PropertySchema(BaseSchema):
    def get_cell_type(self, node_params):
        cell_type = node_params.get('level_of_detail', 'virtual')
        if cell_type == 'biophysical':
            return CellTypes.Biophysical
        elif cell_type == 'intfire':
            return CellTypes.Point
        elif cell_type == 'virtual' or cell_type == 'filter':
            return CellTypes.Virtual
        else:
            return CellTypes.Unknown

    def get_positions(self, node_params):
        # TODO: this needs to be more robust, should check for x/y/z convention
        if 'x_soma' in node_params:
            return np.array([node_params['x_soma'], node_params['y_soma'], node_params['z_soma']])
        else:
            return None

    def get_edge_weight(self, src_node, trg_node, edge):
        # TODO: check to see if weight function is None or non-existant
        weight_fnc = nrn.py_modules.synaptic_weight(edge['weight_function'])
        return weight_fnc(trg_node, src_node, edge)

    def preselected_targets(self):
        return False

    def target_sections(self, edge):
        try:
            return ast.literal_eval(edge['target_sections'])
        except Exception:
            return []

    def target_distance(self, edge):
        try:
            return json.loads(edge['distance_range'])
        except Exception:
            try:
                return float(edge['distance_range'])
            except Exception:
                return float('NaN')

    def nsyns(self, edge):
        if 'nsyns' in edge:
            return edge['nsyns']
        return 1

    def get_params_column(self):
        return 'params_file'

    def get_morphology_column(self, node_params):
        if 'morphology' in node_params:
            return node_params['morphology']
        elif 'morphology_file' in node_params:
            return node_params['morphology_file']
        else:
            raise Exception('Could not find morphology column.')

    def model_type(self, node):
        return node['level_of_detail']

    """
    def load_cell_hobj(self, node):
        model_type_str = PropertySchema.model_type_str(node)
        #model_type_str = node.model_type
        cell_fnc = nrn.py_modules.cell_model(model_type_str)
        return cell_fnc(node)
        #if model_type_str in nrn.py_modules.cell_models:
        #   nrn.py_modules.cell_model()
        # print node.model_params
        #exit()
    """

