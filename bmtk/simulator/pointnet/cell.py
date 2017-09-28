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
import nest


class Cell(object):
    """Generic cell object that contains both NEST object information and non-nest parameters."""

    def __init__(self, node_params):
        self._node_params = node_params  # Most params are not used by nest but can still be accessed
        self._node_id = node_params.node_id
        self._model_type = node_params.model_type  # type of nest model cell build
        self._model_params = node_params.model_params  # dictionary for nest model
        # node_params.dynamics_params

        # build the nest cell
        self._nest_id_list = self._build_cell()
        # self._nest_id_list = nest.Create(self._model_type, 1, self._model_params)
        self._nest_id = self._nest_id_list[0]  # We are building only one object but NEST returns a list

    @property
    def node_id(self):
        return self._node_id

    @property
    def nest_id(self):
        return self._nest_id

    @property
    def nest_id_list(self):
        return self._nest_id_list

    @property
    def model_type(self):
        return self._model_type

    @property
    def model_params(self):
        return self._model_params

    def _build_cell(self):
        raise NotImplementedError()


class NestCell(Cell):
    """Used for internal nest cells, can be any type of valid NEST cell model"""

    def _build_cell(self):
        return nest.Create(self.model_type, 1, self.model_params)

    def set_spike_detector(self, spike_detector):
        nest.Connect(self._nest_id_list, spike_detector)

    def set_synaptic_connection(self, src_cell, trg_cell, edge_props):
        src_id = src_cell.nest_id_list
        trg_id = self.nest_id_list
        syn_model = edge_props['synapse_model']
        syn_dict = edge_props['dynamics_params']
        syn_dict['delay'] = edge_props.delay  # TODO: delay may be in the dynamic params
        syn_dict['weight'] = edge_props.weight(src_cell._node_params, trg_cell._node_params)

        # TODO: don't build the rule every time
        nest.Connect(src_id, trg_id, {'rule': 'all_to_all'}, syn_dict)


class VirtualCell(Cell):
    """Special for external (virtual) cells. For now external cells must be spike_generator type NEST cells."""
    def _build_cell(self):
        return nest.Create(self.model_type, 1, self.model_params)

    @property
    def model_type(self):
        return 'spike_generator'

    def set_spike_train(self, spike_times):
        # TODO: there is issues if the spike times are out-of-order, or if they are not lined up with the given
        #       resolution (dt). Need some further preprocessing.
        if spike_times is None or len(spike_times) == 0:
            return

        if spike_times[0] == 0.0:
            # NEST doesn't allow spikes at time 0 which some of our data does have
            spike_times = spike_times[1:]

        nest.SetStatus(self.nest_id_list, {'spike_times': spike_times})
