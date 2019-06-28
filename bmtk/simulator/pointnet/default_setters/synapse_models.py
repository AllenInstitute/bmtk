from bmtk.simulator.pointnet.pyfunction_cache import add_synapse_model


def static_synapse(edge, source_node=None, target_node=None):
    model_params = {
        'model': 'static_synapse',
        'delay': edge.delay,
        'weight': edge.syn_weight(source_node, target_node)
    }

    model_params.update(edge.dynamics_params)
    return model_params


add_synapse_model(static_synapse, 'default', overwrite=False)
add_synapse_model(static_synapse, overwrite=False)