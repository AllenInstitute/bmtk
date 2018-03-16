# Copyright 2017. Allen Institute. All rights reserved
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
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
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

