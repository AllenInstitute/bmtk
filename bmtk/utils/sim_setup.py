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
import os
import shutil
import json
import h5py
import re
from subprocess import call
from optparse import OptionParser
from collections import OrderedDict

# Order of the different sections of the config.json. Any non-listed items will be placed at the end of the config
config_order = [
    'manifest',
    'target_simulator',
    'run',
    'conditions',
    'inputs',
    'components',
    'output',
    'reports',
    'networks'
]

local_path = os.path.dirname(os.path.realpath(__file__))
scripts_path = os.path.join(local_path, 'scripts')

'''
order_lookup = {k: i for i, k in enumerate(config_order)}
def sort_config_keys(ckey):
    print(ckey)
    exit()
'''

def get_network_block(circuit_config, network_dir):
    net_nodes = {}
    net_edges = {}
    for f in os.listdir(network_dir):
        if not os.path.isfile(os.path.join(network_dir, f)) or f.startswith('.'):
            continue

        if '_nodes' in f:
            net_name = f[:f.find('_nodes')]
            nodes_dict = net_nodes.get(net_name, {})
            nodes_dict['nodes_file'] = os.path.join('$NETWORK_DIR', f)
            net_nodes[net_name] = nodes_dict

        elif '_node_types' in f:
            net_name = f[:f.find('_node_types')]
            nodes_dict = net_nodes.get(net_name, {})
            nodes_dict['node_types_file'] = os.path.join('$NETWORK_DIR', f)
            net_nodes[net_name] = nodes_dict

        elif '_edges' in f:
            net_name = f[:f.find('_edges')]
            edges_dict = net_edges.get(net_name, {})
            edges_dict['edges_file'] = os.path.join('$NETWORK_DIR', f)
            try:
                edges_h5 = h5py.File(os.path.join(network_dir, f), 'r')
                edges_dict['target'] = edges_h5['edges']['target_gid'].attrs['network']
                edges_dict['source'] = edges_h5['edges']['source_gid'].attrs['network']
            except Exception as e:
                pass

            net_edges[net_name] = edges_dict

        elif '_edge_types' in f:
            net_name = f[:f.find('_edge_types')]
            edges_dict = net_edges.get(net_name, {})
            edges_dict['edge_types_file'] = os.path.join('$NETWORK_DIR', f)
            net_edges[net_name] = edges_dict

        else:
            print('Unknown file {}. Will have to enter by hand'.format(f))

    for _, sect in net_nodes.items():
        circuit_config['networks']['nodes'].append(sect)

    for _, sect in net_edges.items():
        circuit_config['networks']['edges'].append(sect)


def build_components(circuit_config, components_path, scripts_path, with_examples):
    for c_name, c_dir in circuit_config['components'].items():
        dir_name = c_dir.replace('$COMPONENTS_DIR/', '')
        dir_path = os.path.join(components_path, dir_name)

        # create component directory
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Copy in files from scripts/<simulator>/<component_dir>
        scripts_dir = os.path.join(scripts_path, dir_name)
        if with_examples and os.path.isdir(scripts_dir):
            shutil.rmtree(dir_path)
            shutil.copytree(scripts_dir, dir_path)


def build_circuit_env(base_dir, network_dir, components_dir, simulator, with_examples):
    simulator_path = os.path.join(scripts_path, simulator)

    circuit_config = json.load(open(os.path.join(scripts_path, 'sonata.circuit_config.json')))
    circuit_config['manifest']['$BASE_DIR'] = base_dir if base_dir == '.' else os.path.abspath(base_dir)
    circuit_config['manifest']['$COMPONENTS_DIR'] = '$BASE_DIR/{}'.format(components_dir)

    # Try to figure out the $NETWORK_DIR
    if network_dir is None:
        network_path = ''
    if os.path.isabs(network_dir):
        # In case network_dir is an absolute path
        network_path = network_dir
    elif os.path.abspath(network_dir).startswith(os.path.abspath(base_dir)):
        # If network_dir is in a subdir of base_dir then NETWORK_DIR=$BASE_DIR/path/to/network
        network_path = os.path.abspath(network_dir).replace(os.path.abspath(base_dir), '$BASE_DIR')
    else:
        # if network_dir exists outside of the base_dir just reference the absolute path
        network_path = os.path.abspath(network_dir)

    circuit_config['manifest']['$NETWORK_DIR'] = network_path

    # Initialize the components directories
    build_components(circuit_config, os.path.join(base_dir, components_dir), simulator_path, with_examples)

    # Parse the network directory
    get_network_block(circuit_config, network_dir)

    return circuit_config


def build_simulation_env(base_dir, target_simulator, tstop, dt, reports):
    simulation_config = json.load(open(os.path.join(scripts_path, 'sonata.simulation_config.json')))
    simulation_config['manifest']['$BASE_DIR'] = base_dir if base_dir == '.' else os.path.abspath(base_dir)
    simulation_config['target_simulator'] = target_simulator
    simulation_config['run']['tstop'] = tstop
    simulation_config['run']['dt'] = dt

    if reports is not None:
        for report_name, report_params in reports.items():
            simulation_config['reports'][report_name] = report_params

    return simulation_config


def copy_config(base_dir, json_dict, config_file_name):
    with open(os.path.join(base_dir, config_file_name), 'w') as outfile:
        ordered_dict = OrderedDict(sorted(json_dict.items(),
                                          key=lambda s: config_order.index(s[0]) if s[0] in config_order else 100))
        json.dump(ordered_dict, outfile, indent=2)


def copy_run_script(base_dir, simulator, run_script):
    simulator_path = os.path.join(scripts_path, simulator)
    shutil.copy(os.path.join(simulator_path, run_script), os.path.join(base_dir, run_script))


def build_env_pointnet(base_dir='.', network_dir=None, reports=None, with_examples=True, tstop=1000.0, dt=0.001, **args):
    simulator='pointnet'
    target_simulator='NEST'
    components_dir='point_components'

    # Copy run script
    copy_run_script(base_dir=base_dir, simulator=simulator, run_script='run_{}.py'.format(simulator))

    # Build circuit_config and componenets directory
    circuit_config = build_circuit_env(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
                                       simulator=simulator, with_examples=with_examples)
    copy_config(base_dir, circuit_config, 'circuit_config.json')

    simulation_config = build_simulation_env(base_dir=base_dir, target_simulator=target_simulator, tstop=tstop, dt=dt,
                                             reports=reports)
    copy_config(base_dir, simulation_config, 'simulation_config.json')


def build_env_bionet(base_dir='.', network_dir=None, reports=None, with_examples=True, tstop=1000.0, dt=0.001,
                     compile_mechanisms=True, **args):
    simulator='bionet'
    target_simulator='NEURON'
    components_dir='biophys_components'

    # Copy run script
    copy_run_script(base_dir=base_dir, simulator=simulator, run_script='run_{}.py'.format(simulator))

    # Build circuit_config and componenets directory
    circuit_config = build_circuit_env(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
                                       simulator=simulator, with_examples=with_examples)
    copy_config(base_dir, circuit_config, 'circuit_config.json')
    if compile_mechanisms:
        cwd = os.getcwd()
        os.chdir(circuit_config['components']['mechanisms_dir'])
        try:
            print(os.getcwd())
            call(['nrnivmodl', 'modfiles'])
        except Exception as e:
            print('Was unable to compile mechanism in {}'.format(circuit_config['components']['mechanisms_dir']))
            # print e.message
        os.chdir(cwd)

    # Build simulation config
    simulation_config = build_simulation_env(base_dir=base_dir, target_simulator=target_simulator, tstop=tstop, dt=dt,
                                             reports=reports)
    simulation_config['run']['dL'] = args.get('dL', 20.0)
    simulation_config['run']['spike_threshold'] = args.get('spike_threshold', -15.0)
    simulation_config['run']['nsteps_block'] = args.get('nsteps_block', 5000)
    simulation_config['conditions']['v_init'] = args.get('v_init', -80.0)
    copy_config(base_dir, simulation_config, 'simulation_config.json')


def build_env_popnet(base_dir='.', network_dir=None, reports=None, with_examples=True, tstop=1000.0, dt=0.001, **args):
    simulator='popnet'
    target_simulator='DiPDE'
    components_dir='pop_components'

    # Copy run script
    copy_run_script(base_dir=base_dir, simulator=simulator, run_script='run_{}.py'.format(simulator))

    # Build circuit_config and componenets directory
    circuit_config = build_circuit_env(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
                                       simulator=simulator, with_examples=with_examples)
    circuit_config['components']['population_models_dir'] = '$COMPONENTS_DIR/population_models'
    # population_models_dir = os.path.join(base_dir, components_dir, 'population_models')
    if with_examples:
        models_dir =  os.path.join(base_dir, components_dir, 'population_models')
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
        shutil.copytree(os.path.join(scripts_path, simulator, 'population_models'), models_dir)

    copy_config(base_dir, circuit_config, 'circuit_config.json')

    # Build simulation config
    simulation_config = build_simulation_env(base_dir=base_dir, target_simulator=target_simulator, tstop=tstop, dt=dt,
                                             reports=reports)
    # PopNet doesn't produce spike files so instead need to replace them with rates files
    for output_key in simulation_config['output'].keys():
        if output_key.startswith('spikes'):
            del simulation_config['output'][output_key]
    # simulation_config['output']['rates_file_csv'] = 'firing_rates.csv'
    simulation_config['output']['rates_file'] = 'firing_rates.csv'

    copy_config(base_dir, simulation_config, 'simulation_config.json')


"""
def build_env_bionet(base_dir='.', run_time=0.0, with_config=True, network_dir=None, with_cell_types=True,
                     compile_mechanisms=True, reports=None):
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
                print(os.getcwd())
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

        if reports is not None:
            for report_name, report_params in reports.items():
                config_json['reports'][report_name] = report_params

        ordered_dict = OrderedDict(sorted(config_json.items(),
                                          key=lambda s: config_order.index(s[0]) if s[0] in config_order else 100))
        with open(os.path.join(base_dir, 'config.json'), 'w') as outfile:
            json.dump(ordered_dict, outfile, indent=2)
            #json.dump(config_json, outfile, indent=2)
"""


if __name__ == '__main__':
    def str_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    #def int_list(option, opt, value, parser):
    #    setattr(parser.values, option.dest, [int(v) for v in value.split(',')])

    def parse_node_set(option, opt, value, parser):
        try:
            setattr(parser.values, option.dest, [int(v) for v in value.split(',')])
        except ValueError as ve:
            setattr(parser.values, option.dest, value)


    parser = OptionParser(usage="Usage: python -m bmtk.utils.sim_setup [options] bionet|pointnet|popnet|mintnet")
    parser.add_option('-b', '--base_dir', dest='base_dir', default='.', help='path of environment')
    parser.add_option('-n', '--network_dir', dest='network_dir', default=None,
                      help="Use an exsting directory with network files.")
    parser.add_option('-r', '--tstop', type='float', dest='tstop', default=1000.0)
    parser.add_option('-d', '--dt', type=float, dest='dt', help='simulation time step dt', default=0.001)

    # For membrane report
    def membrane_report_parser(option, opt, value, parser):
        parser.values.has_membrane_report = True
        if ',' in value:
            try:
                setattr(parser.values, option.dest, [int(v) for v in value.split(',')])
            except ValueError as ve:
                setattr(parser.values, option.dest, value.split(','))

        else:
            setattr(parser.values, option.dest, value)

    parser.add_option('--membrane_report', dest='has_membrane_report', action='store_true', default=False)
    parser.add_option('--membrane_report-vars', dest='mem_rep_vars', type='string', action='callback',
                      callback=membrane_report_parser, default=[])
    parser.add_option('--membrane_report-cells', dest='mem_rep_cells', type='string', action='callback',
                      callback=membrane_report_parser, default='all')
    # parser.add_option('--membrane_report_file', dest='mem_rep_file', type='string', action='callback',
    #                   callback=membrane_report_parser, default='$OUTPUT_DIR/cell_vars.h5')
    parser.add_option('--membrane_report-sections', dest='mem_rep_secs', type='string', action='callback',
                      callback=membrane_report_parser, default='all')

    options, args = parser.parse_args()
    reports = {}

    if options.has_membrane_report:
        reports['membrane_report'] = {
            'module': 'membrane_report',
            'variable_name': options.mem_rep_vars,
            'cells': options.mem_rep_cells,
            # 'file_name': options.mem_rep_file,
            'sections': options.mem_rep_secs,
        }

    target_sim = args[0].lower() if len(args) == 1 else None
    if target_sim not in ['bionet', 'popnet', 'pointnet', 'mintnet']:
        raise Exception('Must specify one target simulator. options: "bionet", pointnet", "popnet" or "mintnet"')

    if target_sim == 'bionet':
        build_env_bionet(base_dir=options.base_dir, network_dir=options.network_dir, tstop=options.tstop,
                         dt=options.dt, reports=reports)

    elif target_sim == 'pointnet':
        build_env_pointnet(base_dir=options.base_dir, network_dir=options.network_dir, tstop=options.tstop,
                           dt=options.dt, reports=reports)

    elif target_sim == 'popnet':
        build_env_popnet(base_dir=options.base_dir, network_dir=options.network_dir, tstop=options.tstop,
                           dt=options.dt, reports=reports)
