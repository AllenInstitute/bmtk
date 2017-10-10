# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the Allen Institute's written permission. For
# purposes of this license, commercial purposes is the incorporation of the Allen Institute's software into anything for
# which you will charge fees or other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
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

        elif 'positions' in node_params:
            return node_params['positions']

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
        return node['set_params_function']
        #return node['level_of_detail']

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

