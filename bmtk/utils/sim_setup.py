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
import sys
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
    logger.info('Creating config file: {}'.format(config_file_name))
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
                logger.info('  Adding edge types file: {}'.format(edges_dict['edge_types_file']))

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


def __set_manifest(config_dict, base_dir, network_dir=None, components_dir=None, output_dir=None):
    config_dict['manifest'] = {'$BASE_DIR': '${configdir}'}
    base_dir = os.path.abspath(base_dir)

    replace_str = lambda fd, bd, var_name: fd.replace(bd, var_name) if fd.startswith(bd) else fd

    if network_dir is not None:
        config_dict['manifest']['$NETWORK_DIR'] = replace_str(network_dir, base_dir, '$BASE_DIR')
        if len(config_dict['networks'].get('nodes', [])) > 0:
            #config_dict['networks']['nodes'] = [{k: replace_str(v, network_dir, '$NETWORK_DIR')
            #                                     for l in config_dict['networks']['nodes'] for k, v in l.items()}]
            config_dict['networks']['nodes'] = [{k: replace_str(v, network_dir, '$NETWORK_DIR')
                                                 for k, v in l.items()} for l in config_dict['networks']['nodes'] ]


        if len(config_dict['networks'].get('edges', [])) > 0:
            #config_dict['networks']['edges'] = [{k: replace_str(v, network_dir, '$NETWORK_DIR')
            #                                     for l in config_dict['networks']['edges'] for k, v in l.items()}]
            config_dict['networks']['edges'] = [{k: replace_str(v, network_dir, '$NETWORK_DIR')
                                                 for k, v in l.items()} for l in config_dict['networks']['edges'] ]

    if components_dir is not None:
        config_dict['manifest']['$COMPONENTS_DIR'] = replace_str(components_dir, base_dir, '$BASE_DIR')
        for k, v in config_dict['components'].items():
            config_dict['components'][k] = replace_str(v, components_dir, '$COMPONENTS_DIR')

    if output_dir is not None:
        config_dict['manifest']['$OUTPUT_DIR'] = os.path.join('$BASE_DIR', output_dir)

    if 'inputs' in config_dict:
        for input_dict in config_dict['inputs'].values():
            if 'input_file' in input_dict:
                input_dict['input_file'] = replace_str(input_dict['input_file'], base_dir, '$BASE_DIR')

            if 'file_name' in input_dict:
                input_dict['file_name'] = replace_str(input_dict['file_name'], base_dir, '$BASE_DIR')


def __create_node_sets_file(base_dir, ns_file_path=None, **custom_ns):
    if ns_file_path is None or not os.path.isabs(ns_file_path):
        abs_path = os.path.abspath(os.path.join(base_dir, ns_file_path or 'node_sets.json'))
    else:
        abs_path = ns_file_path

    if os.path.exists(abs_path):
        logger.info('Found existing node sets file: {}'.format(ns_file_path))
    else:
        logger.info('Creating new node sets file: {}'.format(abs_path))
        node_sets = {
            'biophysical_nodes': {'model_type': 'biophysical'},
            'point_nodes': {'model_type': 'point_process'}
        }
        for k, v in custom_ns.items():
            node_sets[''] = {'node_ids': v} if isinstance(v, list) else {'population': v}

        json.dump(node_sets, open(abs_path, 'w'), indent=2)

    return abs_path


def __add_reports(cell_vars, node_set, section='soma'):
    for v in cell_vars:
        logger.info('Adding membrane report for variable {}'.format(v))

    report_config = {
        '{}_report'.format(v): {
            'variable_name': v,
            'cells': node_set,
            'module': 'membrane_report',
            'sections': section
        } for v in cell_vars}

    return report_config


def __add_output_section():
    return {
        'log_file': 'log.txt',
        'output_dir': '$OUTPUT_DIR',
        'spikes_file': 'spikes.h5'
    }


def __add_current_clamp(amp, delay, duration):
    logger.info('Adding current clamp')
    return {
        "input_type": "current_clamp",
        "module": "IClamp",
        "node_set": "all",
        "amp": float(amp),
        "delay": float(delay),
        "duration": float(duration)
    }

'''
def _initalize_logger():
    logger.setLevel(logging.INFO)
    console_logger = logging.StreamHandler()
    console_logger.setFormatter(logging.Formatter('%(module)s [%(levelname)s] %(message)s'))
    logger.addHandler(console_logger)
'''

def set_logging():
    """"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create STDERR handler
    handler = logging.StreamHandler(sys.stdout)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(module)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    # Set STDERR handler as the only handler
    logger.handlers = [handler]


def __add_spikes_inputs(spikes_inputs):
    inputs_dict = {}
    for s in spikes_inputs:
        pop_name = s[0] or 'all'
        input_name = '{}_spikes'.format(s[0] or 'input')
        spikes_file = os.path.abspath(s[1])
        # if os.path.abspath(spikes_file).startswith()

        spikes_ext = os.path.splitext(spikes_file)[1][1:]
        spikes_ext = 'sonata' if spikes_ext in ['h5', 'hdf5'] else spikes_ext

        inputs_dict[input_name] = {
            "input_type": "spikes",
            "module": spikes_ext,
            "input_file": spikes_file,
            "node_set": pop_name
        }

    return inputs_dict


def build_env_bionet(base_dir='.', network_dir=None, components_dir=None, node_sets_file=None, include_examples=False,
                     tstart=0.0, tstop=1000.0, dt=0.001, dL=20.0, spikes_threshold=-15.0, nsteps_block=5000,
                     v_init=-80.0, celsius=34.0,
                     report_vars=[], report_nodes=None,
                     current_clamp=None,
                     spikes_inputs=None,
                     compile_mechanisms=False,
                     use_relative_paths=True):
    logger.info('Creating BioNet simulation environment ({})'.format(datetime.datetime.now()))
    simulator='bionet'
    target_simulator='NEURON'

    # Create files and json for circuit configurations.
    circuit_config = {}

    parsed_base_dir = __get_base_dir(base_dir)

    parsed_network_dir = __get_network_dir(parsed_base_dir, network_dir)
    circuit_config['networks'] = __parse_network_dir(parsed_network_dir)

    parsed_components_dir = __get_components_dir(base_dir, components_dir)
    circuit_config['components'] = __create_components_dir(parsed_components_dir,
                                                           comp_root=os.path.join(scripts_path, 'bionet'),
                                                           copy_files=include_examples)

    if use_relative_paths:
        __set_manifest(circuit_config, parsed_base_dir, network_dir=parsed_network_dir,
                       components_dir=parsed_components_dir)
    copy_config(base_dir, circuit_config, 'circuit_config.json')


    # Create node sets files and figure out the
    simulation_config = {}
    if report_nodes is not None:
        selected_ns = 'report_nodes'
        abs_nsfile = __create_node_sets_file(base_dir, node_sets_file, report_nodes=report_nodes)
    else:
        selected_ns = 'biophysical_nodes'
        abs_nsfile = __create_node_sets_file(base_dir, node_sets_file)

    simulation_config['node_sets_file'] = abs_nsfile
    simulation_config['reports'] = __add_reports(report_vars, selected_ns)
    simulation_config['output'] = __add_output_section()
    simulation_config['target_simulator'] = target_simulator
    simulation_config['network'] = os.path.join('$BASE_DIR', 'circuit_config.json')
    simulation_config['run'] = {
        'tstart': tstart,
        'tstop': tstop,
        'dt': dt,
        'dL': dL,
        'spike_threshold': spikes_threshold,
        'nsteps_block': nsteps_block
    }
    simulation_config['conditions'] = {
        'celsius': celsius,
        'v_init': v_init
    }

    simulation_config['inputs'] = {}
    if current_clamp is not None:
        simulation_config['inputs']['current_clamp'] = __add_current_clamp(**current_clamp)

    if spikes_inputs is not None and len(spikes_inputs) > 0:
        simulation_config['inputs'].update(__add_spikes_inputs(spikes_inputs))

    __set_manifest(simulation_config, parsed_base_dir, output_dir='output')
    copy_config(base_dir, simulation_config, 'simulation_config.json')

    logger.info('Copying run_bionet.py file')
    copy_run_script(base_dir=base_dir, simulator=simulator, run_script='run_{}.py'.format(simulator))

    if compile_mechanisms:
        mechanisms_dir = os.path.join(parsed_components_dir, 'mechanisms')
        logger.info('Attempting to compile NEURON mechanims under "{}"'.format(mechanisms_dir))
        cwd = os.getcwd()

        try:
            os.chdir(os.path.join(mechanisms_dir))
            call(['nrnivmodl', 'modfiles'])
            logger.info('  Success.')
        except Exception as e:
            logger.error('  Was unable to compile mechanism in {}'.format(mechanisms_dir))
        os.chdir(cwd)


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

    #def str_list(option, opt, value, parser):
    #    setattr(parser.values, option.dest, value.split(','))
    #def int_list(option, opt, value, parser):
    #    setattr(parser.values, option.dest, [int(v) for v in value.split(',')])
    #def parse_node_set(option, opt, value, parser):
    #    try:
    #        setattr(parser.values, option.dest, [int(v) for v in value.split(',')])
    #    except ValueError as ve:
    #        setattr(parser.values, option.dest, value)


    parser = OptionParser(usage="Usage: python %prog [options] [bionet|pointnet|popnet|filternet] sim_dir")
    parser.add_option('-n', '--network_dir', dest='network_dir', default=None,
                      help="Use an exsting directory with network files.")
    parser.add_option('--tstop', type='float', dest='tstop', default=1000.0)
    parser.add_option('--dt', type=float, dest='dt', help='simulation time step dt', default=0.001)

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

    parser.add_option('--report-vars', dest='mem_rep_vars', type='string', action='callback',
                      callback=membrane_report_parser, default=[],
                      help='A list of membrane variables to record from; v, cai, etc.')
    parser.add_option('--report-nodes', dest='mem_rep_cells', type='string', action='callback',
                      callback=membrane_report_parser, default=None)
    parser.add_option('--iclamp', dest='current_clamp', type='string', action='callback',
                      callback=membrane_report_parser, default=None,
                      help='Adds a soma current clamp using three variables: <amp>,<delay>,<duration> (nA, ms, ms)')
    parser.add_option('--spikes-inputs', dest='spikes_input', type='string', action='callback',
                      callback=membrane_report_parser, default=None)
    # parser.add_option('--membrane_report_file', dest='mem_rep_file', type='string', action='callback',
    #                   callback=membrane_report_parser, default='$OUTPUT_DIR/cell_vars.h5')
    #parser.add_option('--membrane_report-sections', dest='mem_rep_secs', type='string', action='callback',
    #                  callback=membrane_report_parser, default='all')

    parser.add_option('--include-examples', dest='include_examples', action='store_true', default=False,
                      help='Copies component files used by examples and tutorials.')
    parser.add_option('--compile-mechanisms', dest='compile_mechanisms', action='store_true', default=False,
                      help='Will try to compile the NEURON mechanisms (BioNet only).')


    options, args = parser.parse_args()

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

    if options.current_clamp is not None:
        cc_args = options.current_clamp
        if len(cc_args) != 3:
            parser.error('Invalid arguments for current clamp, requires three floating point numbers '
                         '<ampage>,<delay>,<duration> (nA, ms, ms)')
        iclamp_args = {'amp': cc_args[0], 'delay': cc_args[1], 'duration': cc_args[2]}
    else:
        iclamp_args = None

    spikes_inputs = []
    if options.spikes_input is not None:
        spikes = [options.spikes_input] if isinstance(options.spikes_input, str) else list(options.spikes_input)
        for spike_str in spikes:
            vals = spike_str.split(':')
            if len(vals) == 1:
                spikes_inputs.append((None, vals[0]))
            elif len(vals) == 2:
                spikes_inputs.append((vals[0], vals[1]))
            else:
                parser.error('Cannot parse spike-input string <pop1>:<spikes-file1>,<pop2>:<spikes-file2>,...')

    if target_sim == 'bionet':
        build_env_bionet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                         dt=options.dt, report_vars=options.mem_rep_vars, report_nodes=options.mem_rep_cells,
                         current_clamp=iclamp_args, include_examples=options.include_examples,
                         spikes_inputs=spikes_inputs,
                         compile_mechanisms=options.compile_mechanisms)

    elif target_sim == 'pointnet':
        build_env_pointnet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                           dt=options.dt, reports=reports)

    elif target_sim == 'popnet':
        build_env_popnet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                           dt=options.dt, reports=reports)
