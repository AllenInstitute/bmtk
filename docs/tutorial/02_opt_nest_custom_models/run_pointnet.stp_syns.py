import os, sys
# import nest

from bmtk.simulator import pointnet
# from bmtk.simulator.pointnet import cell_model
# from bmtk.simulator.pointnet.io_tools import io


# @cell_model(directive='nest:izhikevich', model_type='point_neuron')
# def loadIzhikevich(cell, template_name, dynamics_params):
#     nodes = nest.Create(template_name, cell.n_nodes, dynamics_params)
    
#     d_orig = nest.GetStatus(nodes, 'd')
#     jitter = cell['jitter']
#     d_noisy = d_orig*jitter
#     nest.SetStatus(nodes, {'d': d_noisy})
#     io.log_info(f'Modifying the parameters of {cell.n_nodes} {template_name} neurons.')
#     return nodes


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    if sys.argv[-1] != __file__:
        run(sys.argv[-1])
    else:
        run('config.stp_syns.json')
