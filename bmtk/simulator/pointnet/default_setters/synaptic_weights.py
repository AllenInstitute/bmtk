from bmtk.simulator.pointnet.pyfunction_cache import add_weight_function


def default_weight_fnc(edge_props, source_node, target_node):
    return edge_props['syn_weight']*edge_props.nsyns


add_weight_function(default_weight_fnc, 'default_weight_fnc', overwrite=False)
