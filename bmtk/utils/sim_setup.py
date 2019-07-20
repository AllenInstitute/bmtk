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
import datetime
import logging
from distutils.dir_util import copy_tree
import glob

logger = logging.getLogger(__name__)

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


network_dir_synonyms = ['network', 'networks', 'circuit', 'circuits', 'network_dir',  'circuit_dir']


def __get_base_dir(base_dir):
    # Create base-dir, or if it exists make sure
    if not os.path.exists(base_dir):
        logger.info('Creating directory "{}"'.format(os.path.abspath(base_dir)))
        os.mkdir(base_dir)
    elif os.path.isfile(base_dir):
        logging.error('simulation directory path {} points to an existing file, cannot create.'.format(base_dir))
        exit(1)
    else:
        logging.info('Using existing directory "{}"'.format(os.path.abspath(base_dir)))

    return os.path.abspath(base_dir)


def __get_network_dir(base_dir, network_dir=None):
    """Attempts to get the appropiate path the the directory containing the sonata network files, either as specified
    by the user (-n <network_dir> argument) or else attempt to find the location of the network files.

    :param base_dir:
    :param network_dir:
    :return: The absolute path of the directory that does/should contain the network files
    """
    network_dir_abs = None
    print(base_dir)

    if network_dir is None:
        # Check to see if there are any folders in base_dir that might contain SONATA network files
        sub_dirs = [os.path.join(base_dir, dn) for dn in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, dn))
                    and dn.lower() in network_dir_synonyms]
        if sub_dirs:
            # See if there are any subfolders that might contain network files
            network_dir_abs = os.path.abspath(sub_dirs[0])
            logging.info('No network folder specified, attempting to use existing folder: {}'.format(network_dir_abs))
        else:
            network_dir_abs = os.path.abspath(os.path.join(base_dir, 'network'))
            logging.info('No network folder specified, creating empty folder: {}'.format(network_dir_abs))
            os.makedirs(network_dir_abs)

    elif os.path.isabs(network_dir):
        # In the case the user specifies the network directory as an absolute path
        network_dir_abs = network_dir
        logging.info('Using network directory: {}'.format(network_dir_abs))

    elif os.path.isdir(network_dir):
        # In case during calling the script the user was pointing to an already existing directory
        network_dir_abs = os.path.abspath(network_dir)
        logging.info('Using network directory: {}'.format(network_dir_abs))

    else:
        # Default case is the to use/add the <network_dir> found under base_dir
        network_dir_abs = os.path.join(base_dir, network_dir)
        if not os.path.exists(network_dir_abs):
            logging.info('Creating new network directory: {}'.format(network_dir_abs))
            os.makedirs(network_dir_abs)
        else:
            logging.info('Using network directory: {}'.format(network_dir_abs))

    return network_dir_abs


def __parse_network_dir(network_dir):
    logger.info('Parsing {} for SONATA network files'.format(network_dir))
    net_nodes = {}
    net_edges = {}
    for root, dirs, files in os.walk(network_dir):
        for f in files:
            if not os.path.isfile(os.path.join(network_dir, f)) or f.startswith('.'):
                continue

            if '_nodes' in f:
                net_name = f[:f.find('_nodes')]
                nodes_dict = net_nodes.get(net_name, {})
                nodes_dict['nodes_file'] = os.path.abspath(os.path.join(root, f))
                net_nodes[net_name] = nodes_dict
                logger.info('  Adding nodes file: {}'.format(nodes_dict['nodes_file']))

            elif '_node_types' in f:
                net_name = f[:f.find('_node_types')]
                nodes_dict = net_nodes.get(net_name, {})
                nodes_dict['node_types_file'] = os.path.abspath(os.path.join(root, f))
                net_nodes[net_name] = nodes_dict
                logger.info('  Adding node types file: {}'.format(nodes_dict['node_types_file']))

            elif '_edges' in f:
                net_name = f[:f.find('_edges')]
                edges_dict = net_edges.get(net_name, {})
                edges_dict['edges_file'] = os.path.abspath(os.path.join(root, f))
                net_edges[net_name] = edges_dict
                logger.info('  Adding edges file: {}'.format(edges_dict['edges_file']))

            elif '_edge_types' in f:
                net_name = f[:f.find('_edge_types')]
                edges_dict = net_edges.get(net_name, {})
                edges_dict['edge_types_file'] = os.path.abspath(os.path.join(root, f))
                net_edges[net_name] = edges_dict
                logger.info('  Adding edge types file: {}'.format(edges_dict['edges_file']))

            else:
                logger.info('  Skipping file (could not categorize): {}'.format(os.path.abspath(os.path.join(root, f))))

    if not (net_nodes or net_edges):
        logger.info('  Could not find any sonata nodes or edges file(s).')

    network_config = {'nodes': [], 'edges': []}
    for _, sect in net_nodes.items():
        network_config['nodes'].append(sect)

    for _, sect in net_edges.items():
        network_config['edges'].append(sect)

    return network_config


def __get_components_dir(base_dir, components_dir=None):
    if components_dir is None:
        return os.path.abspath(os.path.join(base_dir, 'components'))

    elif os.path.isabs(components_dir):
        return components_dir

    elif os.path.exists(components_dir):
        return os.path.abspath(components_dir)

    else:
        return os.path.abspath(os.path.join(base_dir, components_dir))


def __create_components_dir(components_dir, comp_root, copy_files=True):
    if not os.path.exists(components_dir):
        logger.info('Creating components directory: {}'.format(components_dir))
        os.makedirs(components_dir)

    components_config = {}

    comps_dirs = [sd for sd in os.listdir(comp_root) if os.path.isdir(os.path.join(comp_root, sd))]
    print(components_dir)
    for sub_dir in comps_dirs:
        comp_name = sub_dir + '_dir'
        src_dir = os.path.join(comp_root, sub_dir)
        trg_dir = os.path.join(components_dir, sub_dir)
        if not os.path.exists(trg_dir):
            logger.info('Creating new components directory: {}'.format(trg_dir))
            os.makedirs(trg_dir)
        else:
            logger.info('Using components directory: {}'.format(trg_dir))

        components_config[comp_name] = trg_dir

        if copy_files:
            logger.info('  Copying files from {}.'.format(src_dir))
            copy_tree(src_dir, trg_dir)

    return components_config

def build_env_bionet(base_dir='.', network_dir=None, components_dir=None,reports=None, with_examples=True, tstop=1000.0,
                     dt=0.001, compile_mechanisms=True, **args):
    logger.info('Creating BioNet simulation environment ({})'.format(datetime.datetime.now()))
    simulator='bionet'
    target_simulator='NEURON'
    # components_dir='biophys_components'

    circuit_config = {}

    parsed_base_dir = __get_base_dir(base_dir)

    parsed_network_dir = __get_network_dir(parsed_base_dir, network_dir)
    circuit_config['networks'] = __parse_network_dir(parsed_network_dir)

    parsed_components_dir = __get_components_dir(base_dir, components_dir)
    circuit_config['components'] = __create_components_dir(parsed_components_dir,
                                                           comp_root=os.path.join(scripts_path, 'bionet'),
                                                           copy_files=with_examples)

    exit()


    # Copy run script
    copy_run_script(base_dir=base_dir, simulator=simulator, run_script='run_{}.py'.format(simulator))

    # Build circuit_config and componenets directory
    circuit_config = build_circuit_env(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
                                       simulator=simulator, with_examples=with_examples)
    copy_config(base_dir, circuit_config, 'circuit_config.json')
    if compile_mechanisms:
        cwd = os.getcwd()
        os.chdir(os.path.join(base_dir, components_dir, 'mechanisms'))  # circuit_config['components']['mechanisms_dir'])
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(module)s [%(levelname)s] %(message)s')
    #logger.setLevel(level=logging.INFO)
    # logger.setFormatter('%(asctime)s [%(levelname)s] %(message)s')

    def str_list(option, opt, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    #def int_list(option, opt, value, parser):
    #    setattr(parser.values, option.dest, [int(v) for v in value.split(',')])

    def parse_node_set(option, opt, value, parser):
        try:
            setattr(parser.values, option.dest, [int(v) for v in value.split(',')])
        except ValueError as ve:
            setattr(parser.values, option.dest, value)


    parser = OptionParser(usage="Usage: python %prog [options] [bionet|pointnet|popnet|filternet] sim_dir")
    # parser.add_option('-b', '--base_dir', dest='base_dir', default='.', help='path of environment')
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

    # Check the passed in argments are correct. [sim] </path/to/dir/>
    if len(args) < 2:
        parser.error('Invalid number of arguments, Please specify a target simulation (bionet, pointnet, filternet,'
                     'popnet) and the path to save the simulation environment.')
    elif len(args) > 2:
        parser.error('Unrecognized arguments {}'.format(args[2:]))
    else:
        target_sim = args[0].lower()
        if target_sim not in ['bionet', 'popnet', 'pointnet', 'mintnet']:
            parser.error('Must specify one target simulator. options: "bionet", pointnet", "popnet", "filternet"')
        base_dir = args[1]

    if target_sim == 'bionet':
        build_env_bionet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                         dt=options.dt, reports=reports)

    elif target_sim == 'pointnet':
        build_env_pointnet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                           dt=options.dt, reports=reports)

    elif target_sim == 'popnet':
        build_env_popnet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                           dt=options.dt, reports=reports)
