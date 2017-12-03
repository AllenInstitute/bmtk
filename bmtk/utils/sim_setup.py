import os
import shutil
import json
import re
from subprocess import call
from optparse import OptionParser


def build_env_bionet(base_dir='.', with_config=True, network_dir=None, with_cell_types=True):
    local_path = os.path.dirname(os.path.realpath(__file__))
    scripts_path = os.path.join(local_path, 'scripts', 'bionet')

    components_dir = os.path.join(base_dir, 'components')
    component_paths = {
        'morphologies_dir': os.path.join(components_dir, 'biophysical', 'morphologies'),
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
        cwd = os.getcwd()
        os.chdir(component_paths['mechanisms_dir'])
        try:
            pass
            # call(['nrnivmodl', 'modfiles'])
        except Exception as e:
            print('Was unable to compile mechanism in {}'.format(component_paths['mechanisms_dir']))
            e.message
        os.chdir(cwd)

        shutil.copy(os.path.join(scripts_path, 'run_bionet.py'), os.path.join(base_dir, 'run_bionet.py'))

    if with_config:
        config_json = json.load(open(os.path.join(scripts_path, 'default_config.json')))

        if network_dir is not None:
            network_fullpath = os.path.abspath(network_dir)

            net_nodes = {}
            for f in os.listdir(network_dir):
                if not os.path.isfile(os.path.join(network_dir, f)) or f.startswith('.'):
                    continue

                if '_nodes' in f:
                    net_name = f[:f.find('_nodes')]
                    nodes_dict = net_nodes.get(net_name, {})
                    nodes_dict['nodes'] = os.path.join('${NETWORK_DIR}', f)
                    net_nodes[net_name] = nodes_dict

                elif '_node_types' in f:
                    net_name = f[:f.find('_node_types')]
                    nodes_dict = net_nodes.get(net_name, {})
                    nodes_dict['node_types'] = os.path.join('${NETWORK_DIR}', f)
                    net_nodes[net_name] = nodes_dict

                else:
                    print('Unknown file {}. Will have to enter by hand'.format(f))

            print net_nodes



if __name__ == '__main__':
    parser = OptionParser(usage="Usage: python -m bmtk.utils.sim_setup [options] bionet|pointnet|popnet|mintnet")
    parser.add_option('-b', '--base_dir', dest='base_dir', default='.', help='path of environment')
    parser.add_option('--network_dir', '-n', dest='network_dir', default=None,
                      help="Use an exsting directory with network files.")
    #parser.add_option('--create_config', '-c', dest='create_config', action='store_true', default=False)
    #parser.add_option('--cell_types_mechanisms', '-a', dest='use_ctdb_mech', action=)

    #parser.add_option("--connections", dest="connections_h5", default=None)
    #parser.add_option("--spikes", dest='spikes_nwb', default=None)
    #parser.add_option("--trial", dest='trial_id', default=None)

    options, args = parser.parse_args()
    target_sim = args[0].lower() if len(args) == 1 else None
    if target_sim not in ['bionet', 'popnet', 'pointnet', 'mintnet']:
        raise Exception('Must specify one target simulator. options: "bionet", pointnet", "popnet" or "mintnet"')

    if target_sim == 'bionet':
        build_env_bionet(**vars(options))

    # build_env_bionet(options.cells_csv, options.connections_h5, options.spikes_nwb, options.trial_id)