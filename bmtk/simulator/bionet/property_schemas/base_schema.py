# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 20XX. Allen Institute. All rights reserved.
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
from bmtk.simulator.bionet import nrn

class CellTypes:
    Biophysical = 0
    Point = 1
    Virtual = 2
    Unknown = 3

    @staticmethod
    def len():
        return 4


class PropertySchema(object):

    #######################################
    # For nodes/cells properties
    #######################################
    def get_cell_type(self, node_params):
        raise NotImplementedError()

    def get_positions(self, node_params):
        raise NotImplementedError()

    def model_type(self, node):
        raise NotImplementedError()

    def get_params_column(self):
        raise NotImplementedError()

    def load_cell_hobj(self, node):
        model_type = self.model_type(node)
        cell_fnc = nrn.py_modules.cell_model(model_type)
        return cell_fnc(node)

    #######################################
    # For edge/synapse properties
    #######################################
    def get_edge_weight(self, src_node, trg_node, edge):
        raise NotImplementedError()

    def preselected_targets(self):
        raise NotImplementedError()

    def target_sections(self, edge):
        raise NotImplementedError()

    def target_distance(self, edge):
        raise NotImplementedError()

    def nsyns(self, edge):
        raise NotImplementedError()

    def load_synapse_obj(self, edge, section_x, section_id):
        synapse_fnc = nrn.py_modules.synapse_model(edge['set_params_function'])
        return synapse_fnc(edge['dynamics_params'], section_x, section_id)
