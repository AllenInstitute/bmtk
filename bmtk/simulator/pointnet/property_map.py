class EdgePropertyMap(object):
    def __init__(self, graph):
        self._graph = graph

    def load_synapse_obj(self, edge, section_x, section_id):
        synapse_fnc = nrn.py_modules.synapse_model(edge['model_template'])
        return synapse_fnc(edge['dynamics_params'], section_x, section_id)

    @classmethod
    def build_map(cls, edge_group, biograph):
        prop_map = cls(biograph)

        # For fetching/calculating synaptic weights
        if 'weight_function' in edge_group.all_columns:
            # Customized function for user to calculate the synaptic weight
            prop_map.syn_weight = types.MethodType(weight_function, prop_map)
        elif 'syn_weight' in edge_group.all_columns:
            # Just return the synaptic weight
            prop_map.syn_weight = types.MethodType(syn_weight, prop_map)
        else:
            io.log_exception('Could not find syn_weight or weight_function properties. Cannot create connections.')

        # For determining the synapse placement
        if 'sec_id' in edge_group.all_columns:
            prop_map.preselected_targets = True
            prop_map.nsyns = types.MethodType(no_nsyns, prop_map)
        elif 'nsyns' in edge_group.all_columns:
            prop_map.preselected_targets = False
            prop_map.nsyns = types.MethodType(nsyns, prop_map)
        else:
            # It will get here for connections onto point neurons
            prop_map.preselected_targets = True
            prop_map.nsyns = types.MethodType(no_nsyns, prop_map)

        # For target sections
        '''
        if 'syn_weight' not in edge_group.all_columns:
            io.log_exception('Edges {} missing syn_weight property for connections.'.format(edge_group.parent.name))
        else:
            prop_map.syn_weight = types.MethodType(syn_weight, prop_map)



        if 'syn_weight' in edge_group.columns:
            prop_map.weight = types.MethodType(syn_weight, prop_map)
            prop_map.preselected_targets = True
            prop_map.nsyns = types.MethodType(no_nsyns, prop_map)
        else:
            prop_map.preselected_targets = False
        '''
        return prop_map