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
