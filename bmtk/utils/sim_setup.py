import os
import shutil
import json
import h5py
import re
from subprocess import call
from optparse import OptionParser


def build_env_bionet(base_dir='.', with_config=True, network_dir=None, with_cell_types=True, cell_vars=[],
                     cell_vars_nodes=[], compile_mechanisms=True, run_time=0.0):
    local_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = os.path.join(local_path, 'scripts', 'bionet')

    components_dir = os.path.join(base_dir, 'components')
    component_paths = {
        'morphologies_dir': os.path.join(components_dir, 'biophysical', 'morphology'),
        'biophysical_models_dir': os.path.join(components_dir, 'biophysical', 'electrophysiology'),
        'mechanisms_dir': os.path.join(components_dir, 'mechanisms'),
        'point_models_dir': os.path.join(components_dir, 'intfire'),
        'synaptic_models_dir': os.path.join(components_dir, 'synaptic_models'),
        'templates_dir': os.path.join(components_dir, 'hoc_templates')
    }
    for path in component_paths.values():
        if not os.path.exists(path):
            os.makedirs(path)

    if with_cell_types:
        shutil.rmtree(component_paths['templates_dir'])
        shutil.copytree(os.path.join(scripts_path, 'hoc_templates'), component_paths['templates_dir'])

        shutil.rmtree(component_paths['mechanisms_dir'])
        shutil.copytree(os.path.join(scripts_path, 'mechanisms'), component_paths['mechanisms_dir'])

        shutil.rmtree(component_paths['synaptic_models_dir'])
        shutil.copytree(os.path.join(scripts_path, 'synaptic_models'), component_paths['synaptic_models_dir'])

        shutil.rmtree(component_paths['point_models_dir'])
        shutil.copytree(os.path.join(scripts_path, 'intfire'), component_paths['point_models_dir'])

        if compile_mechanisms:
            cwd = os.getcwd()
            os.chdir(component_paths['mechanisms_dir'])
            try:
                print os.getcwd()
                call(['nrnivmodl', 'modfiles'])
            except Exception as e:
                print('Was unable to compile mechanism in {}'.format(component_paths['mechanisms_dir']))
                # print e.message
            os.chdir(cwd)

        shutil.copy(os.path.join(scripts_path, 'run_bionet.py'), os.path.join(base_dir, 'run_bionet.py'))

    if with_config:
        config_json = json.load(open(os.path.join(scripts_path, 'default_config.json')))
        config_json['manifest']['$BASE_DIR'] = os.path.abspath(base_dir)
        config_json['manifest']['$COMPONENTS_DIR'] = os.path.join('${BASE_DIR}', 'components')
        config_json['run']['tstop'] = run_time

        if network_dir is not None:
            config_json['manifest']['$NETWORK_DIR'] = os.path.abspath(network_dir)

            net_nodes = {}
            net_edges = {}
            for f in os.listdir(network_dir):
                if not os.path.isfile(os.path.join(network_dir, f)) or f.startswith('.'):
                    continue

                if '_nodes' in f:
                    net_name = f[:f.find('_nodes')]
                    nodes_dict = net_nodes.get(net_name, {'name': net_name})
                    nodes_dict['nodes_file'] = os.path.join('${NETWORK_DIR}', f)
                    net_nodes[net_name] = nodes_dict

                elif '_node_types' in f:
                    net_name = f[:f.find('_node_types')]
                    nodes_dict = net_nodes.get(net_name, {'name': net_name})
                    nodes_dict['node_types_file'] = os.path.join('${NETWORK_DIR}', f)
                    net_nodes[net_name] = nodes_dict

                elif '_edges' in f:
                    net_name = f[:f.find('_edges')]
                    edges_dict = net_edges.get(net_name, {'name': net_name})
                    edges_dict['edges_file'] = os.path.join('${NETWORK_DIR}', f)
                    try:
                        edges_h5 = h5py.File(os.path.join(network_dir, f), 'r')
                        edges_dict['target'] = edges_h5['edges']['target_gid'].attrs['network']
                        edges_dict['source'] = edges_h5['edges']['source_gid'].attrs['network']
                    except Exception as e:
                        pass

                    net_edges[net_name] = edges_dict

                elif '_edge_types' in f:
                    net_name = f[:f.find('_edge_types')]
                    edges_dict = net_edges.get(net_name, {'name': net_name})
                    edges_dict['edge_types_file'] = os.path.join('${NETWORK_DIR}', f)
                    net_edges[net_name] = edges_dict

                else:
                    print('Unknown file {}. Will have to enter by hand'.format(f))

            for _, sect in net_nodes.items():
                config_json['networks']['nodes'].append(sect)

            for _, sect in net_edges.items():
                config_json['networks']['edges'].append(sect)

        if len(cell_vars) > 0:
            config_json['run']['save_cell_vars'] = cell_vars
            config_json['node_id_selections']['save_cell_vars'] = cell_vars_nodes

        with open(os.path.join(base_dir, 'config.json'), 'w') as outfile:
            json.dump(config_json, outfile, indent=2)


if __name__ == '__main__':
    def str_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    def int_list(option, opt, value, parser):
        setattr(parser.values, option.dest, [int(v) for v in value.split(',')])

    parser = OptionParser(usage="Usage: python -m bmtk.utils.sim_setup [options] bionet|pointnet|popnet|mintnet")
    parser.add_option('-b', '--base_dir', dest='base_dir', default='.', help='path of environment')
    parser.add_option('-n', '--network_dir', dest='network_dir', default=None,
                      help="Use an exsting directory with network files.")
    parser.add_option('-v', '--cell-vars', dest='cell_vars', type='string', action='callback', callback=str_list,
                      default=[])
    parser.add_option('-c', '--cell-vars-nodes', dest='cell_vars_nodes', type='string', action='callback',
                      callback=int_list, default=[])
    parser.add_option('-r', '--run-time', type='float', dest='run_time', default=0.0)

    options, args = parser.parse_args()
    target_sim = args[0].lower() if len(args) == 1 else None
    if target_sim not in ['bionet', 'popnet', 'pointnet', 'mintnet']:
        raise Exception('Must specify one target simulator. options: "bionet", pointnet", "popnet" or "mintnet"')

    if target_sim == 'bionet':
        build_env_bionet(**vars(options))
