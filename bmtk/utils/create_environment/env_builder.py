# Copyright 2022. Allen Institute. All rights reserved
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
import json
import six
import numpy as np
from subprocess import call
from collections import OrderedDict
import logging
from distutils.dir_util import copy_tree

from bmtk.utils.compile_mechanisms import copy_modfiles, compile_mechanisms


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

network_dir_synonyms = ['network', 'networks', 'circuit', 'circuits', 'network_dir', 'circuit_dir']


class EnvBuilder(object):
    def __init__(self, base_dir='.', network_dir=None, components_dir=None, output_dir=None, node_sets_file=None):
        self._base_dir = self._get_base_dir(base_dir)
        self._network_dir = self._get_network_dir(network_dir)
        self._components_dir = self._get_components_dir(components_dir)
        self._output_dir = self._get_output_dir(output_dir)

        self._circuit_config = {}
        self._simulation_config = {}

    @property
    def target_simulator(self):
        raise NotImplementedError()

    @property
    def bmtk_simulator(self):
        raise NotImplementedError()

    @property
    def base_dir(self):
        return self._base_dir

    @property
    def scripts_root(self):
        local_path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(local_path, '..', 'scripts')

    @property
    def examples_dir(self):
        raise NotImplementedError()

    @property
    def components_dir(self):
        return self._components_dir

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def network_dir(self):
        return self._network_dir

    def _get_base_dir(self, base_dir):
        """Create base-dir, or if it exists make sure"""
        if not os.path.exists(base_dir):
            logger.info('Creating directory "{}"'.format(os.path.abspath(base_dir)))
            os.mkdir(base_dir)
        elif os.path.isfile(base_dir):
            logging.error('simulation directory path {} points to an existing file, cannot create.'.format(base_dir))
            exit(1)
        else:
            logging.info('Using existing directory "{}"'.format(os.path.abspath(base_dir)))

        return os.path.abspath(base_dir)

    def _get_network_dir(self, network_dir=None):
        """Attempts to get the appropiate path the the directory containing the sonata network files, either as specified
        by the user (-n <network_dir> argument) or else attempt to find the location of the network files.

        :param network_dir:
        :return: The absolute path of the directory that does/should contain the network files
        """
        if network_dir is None:
            # Check to see if there are any folders in base_dir that might contain SONATA network files
            sub_dirs = [os.path.join(self.base_dir, dn) for dn in os.listdir(self.base_dir) if
                        os.path.isdir(os.path.join(self.base_dir, dn))
                        and dn.lower() in network_dir_synonyms]
            if sub_dirs:
                # See if there are any subfolders that might contain network files
                network_dir_abs = os.path.abspath(sub_dirs[0])
                logging.info(
                    'No network folder specified, attempting to use existing folder: {}'.format(network_dir_abs))
            else:
                network_dir_abs = os.path.abspath(os.path.join(self.base_dir, 'network'))
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
            network_dir_abs = os.path.join(self.base_dir, network_dir)
            if not os.path.exists(network_dir_abs):
                logging.info('Creating new network directory: {}'.format(network_dir_abs))
                os.makedirs(network_dir_abs)
            else:
                logging.info('Using network directory: {}'.format(network_dir_abs))

        return network_dir_abs

    def _parse_network_dir(self, network_dir, network_filter=None):
        logger.info('Parsing {} for SONATA network files'.format(network_dir))

        # Create a filter function so that only specific files in network_dir are added to configuration, with rules:
        #   1. If network_filters is None, then all files in network_dir are included
        #   2. If network_filters is a list of files, eg [v1_nodes.h5, v1_node_types.csv, v1_v1_edges.h5, etc] then o
        #      only those files in the list are included in the config.
        #   3. If network_filters is a list of strings, eg [v1], then only files with v1* are included in final list.
        if isinstance(network_filter, six.string_types):
            network_filter = [network_filter]
        def filter_file(f):
            if network_filter is None:
                return True
            for filter_cond in network_filter:
                if filter_cond == f:
                    return True
                elif filter_cond in f:
                    return True

            return False

        net_nodes = {}
        net_edges = {}
        net_gaps = {}
        for root, dirs, files in os.walk(network_dir):
            for f in files:
                if not os.path.isfile(os.path.join(network_dir, f)) or f.startswith('.'):
                    continue

                # Check the filter if the current file
                if not filter_file(f):
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

                elif '_gap_juncs' in f:
                    net_name = f[:f.find('_gap_juncs')]
                    gaps_dict = net_gaps.get(net_name, {})
                    gaps_dict['gap_juncs_file'] = os.path.abspath(os.path.join(root, f))
                    net_gaps[net_name] = gaps_dict
                    logger.info('  Adding gap junctions file: {}'.format(gaps_dict['gap_juncs_file']))

                else:
                    logger.info(
                        '  Skipping file (could not categorize): {}'.format(os.path.abspath(os.path.join(root, f))))

        if not (net_nodes or net_edges):
            logger.info('  Could not find any sonata nodes or edges file(s).')

        network_config = {'nodes': [], 'edges': [], 'gap_juncs': []}
        for _, sect in net_nodes.items():
            network_config['nodes'].append(sect)

        for _, sect in net_edges.items():
            network_config['edges'].append(sect)

        for _, sect in net_gaps.items():
            network_config['gap_juncs'].append(sect)

        self._circuit_config['networks'] = network_config

    def _get_components_dir(self, components_dir=None):
        if components_dir is None:
            return os.path.abspath(os.path.join(self.base_dir, 'components'))

        elif os.path.isabs(components_dir):
            return components_dir

        elif os.path.exists(components_dir):
            return os.path.abspath(components_dir)

        else:
            return os.path.abspath(os.path.join(self.base_dir, components_dir))

    def _create_components_dir(self, components_dir, with_examples=True):
        if not os.path.exists(components_dir):
            logger.info('Creating components directory: {}'.format(components_dir))
            os.makedirs(components_dir)

        components_config = {}
        comps_dirs = [sd for sd in os.listdir(self.examples_dir) if os.path.isdir(os.path.join(self.examples_dir, sd))]
        for sub_dir in comps_dirs:
            comp_name = sub_dir + '_dir'
            src_dir = os.path.join(self.examples_dir, sub_dir)
            trg_dir = os.path.join(components_dir, sub_dir)
            if not os.path.exists(trg_dir):
                logger.info('Creating new components directory: {}'.format(trg_dir))
                os.makedirs(trg_dir)
            else:
                logger.info('Using components directory: {}'.format(trg_dir))

            components_config[comp_name] = trg_dir

            if with_examples:
                logger.info('  Copying files from {}.'.format(src_dir))
                copy_tree(src_dir, trg_dir)

        # return components_config
        self._circuit_config['components'] = components_config

    def _get_output_dir(self, output_dir):
        if output_dir is None:
            return os.path.abspath(os.path.join(self.base_dir, 'output'))

        elif os.path.isabs(output_dir):
            return output_dir

        elif os.path.exists(output_dir):
            return os.path.abspath(output_dir)

        else:
            return os.path.abspath(os.path.join(self.base_dir, output_dir))

    def _add_manifest(self, config_dict, network_dir=None, components_dir=None, output_dir=None):
        # config_dict['manifest'] = {'$BASE_DIR': '${configdir}'}
        config_dict['manifest'] = {'$BASE_DIR': os.path.abspath(self.base_dir)}
        base_dir = os.path.abspath(self.base_dir)

        replace_str = lambda fd, bd, var_name: fd.replace(bd, var_name) if fd.startswith(bd) else fd

        if network_dir is not None:
            config_dict['manifest']['$NETWORK_DIR'] = replace_str(network_dir, base_dir, '$BASE_DIR')
            if len(config_dict['networks'].get('nodes', [])) > 0:
                config_dict['networks']['nodes'] = [
                    {k: replace_str(v, network_dir, '$NETWORK_DIR')
                     for k, v in nodes.items()} for nodes in config_dict['networks']['nodes']
                ]

            if len(config_dict['networks'].get('edges', [])) > 0:
                config_dict['networks']['edges'] = [
                    {k: replace_str(v, network_dir, '$NETWORK_DIR')
                     for k, v in edges.items()} for edges in config_dict['networks']['edges']
                ]

        if 'network' in config_dict and isinstance(config_dict['network'], six.string_types):
            config_dict['network'] = replace_str(config_dict['network'], base_dir, '$BASE_DIR')

        if components_dir is not None:
            config_dict['manifest']['$COMPONENTS_DIR'] = replace_str(components_dir, base_dir, '$BASE_DIR')
            for k, v in config_dict['components'].items():
                config_dict['components'][k] = replace_str(v, components_dir, '$COMPONENTS_DIR')

        if output_dir is not None:
            config_dict['manifest']['$OUTPUT_DIR'] = replace_str(output_dir, base_dir, '$BASE_DIR')
            for k, v in config_dict.get('output', {}).items():
                if isinstance(v, six.string_types):
                    config_dict['output'][k] = replace_str(v, output_dir, '$OUTPUT_DIR')

        for input_dict in config_dict.get('inputs', {}).values():
            if 'input_file' in input_dict:
                input_dict['input_file'] = replace_str(input_dict['input_file'], base_dir, '$BASE_DIR')

            if 'file_name' in input_dict:
                input_dict['file_name'] = replace_str(input_dict['file_name'], base_dir, '$BASE_DIR')

            if 'rates' in input_dict:
                input_dict['rates'] = replace_str(input_dict['rates'], base_dir, '$BASE_DIR')

    def _add_reports(self, cell_vars, node_set, section='soma'):
        if isinstance(cell_vars, six.string_types):
            cell_vars = [cell_vars]

        for v in cell_vars:
            logger.info('Adding membrane report for variable {}'.format(v))

        report_config = {}
        for report_var in cell_vars:
            if isinstance(report_var, (list, tuple)):
                cells = 'all' if report_var[0] is None else {'population': report_var[0]}
                variable_name = report_var[1]
            else:
                cells = 'all'
                variable_name = report_var

            report_name = '{}_report'.format(variable_name)
            report_config[report_name] = {
                'variable_name': variable_name,
                'cells': cells,
                'module': 'membrane_report',
                'sections': section
            }

        if 'reports' not in self._simulation_config:
            self._simulation_config['reports'] = {}

        self._simulation_config['reports'].update(report_config)

    def _add_clamp_reports(self, clamp_reports):
        if isinstance(clamp_reports, six.string_types):
            clamp_reports = [clamp_reports]

        for c in clamp_reports:
            logger.info("Adding clamp report for {} clamp.".format(c))

        report_config = {
            '{}_clamp_report'.format(c): {
                'variable_name': c,
                'module': 'clamp_report'
            } for c in clamp_reports}

        if 'reports' not in self._simulation_config:
            self._simulation_config['reports'] = {}

        self._simulation_config['reports'].update(report_config)

    def _add_output_section(self):
        output_section = {
            'log_file': 'log.txt',
            'output_dir': self.output_dir,
            'spikes_file': 'spikes.h5'
        }

        self._simulation_config['output'] = output_section

    def _add_current_clamp(self, current_param):
        if current_param is None:
            return
        logger.info('Adding current clamp')

        iclamp_config = {
            "input_type": "current_clamp",
            "module": "IClamp",
            "node_set": current_param['node_set'],
            "gids": current_param.get('gids', 'all'),
            "amp": current_param['amp'],
            "delay": current_param['delay'],
            "duration": current_param['duration']
        }

        if 'inputs' not in self._simulation_config:
            self._simulation_config['inputs'] = {}

        self._simulation_config['inputs']['current_clamp'] = iclamp_config

    def _add_file_current_clamp(self, current_param):
        if current_param is None:
            return

        amp_file = os.path.abspath(current_param["input_file"])

        f_iclamp_config = {
            "input_type": "current_clamp",
            "module": "FileIClamp",
            "node_set": "all",
            "input_file": amp_file
        }

        if 'inputs' not in self._simulation_config:
            self._simulation_config['inputs'] = {}

        self._simulation_config['inputs']['file_current_clamp'] = f_iclamp_config

    def _add_se_voltage_clamp(self, clamp_param):
        if clamp_param is None:
            return

        seclamp_config = {
            "input_type": "voltage_clamp",
            "module": "SEClamp",
            "node_set": "all",
            "gids": clamp_param['gids'],
            "amps": clamp_param['amps'],
            "durations": clamp_param["durations"]
        }

        if "rs" in clamp_param.keys():
            seclamp_config["rs"] = clamp_param["rs"]

        if 'inputs' not in self._simulation_config:
            self._simulation_config['inputs'] = {}

        name = 'se_voltage_clamp'
        if 'name' in clamp_param.keys():
            name = clamp_param['name']

        self._simulation_config['inputs'][name] = seclamp_config

    def _add_spikes_inputs(self, spikes_inputs):
        if spikes_inputs is None:
            return

        inputs_dict = {}
        for s in spikes_inputs:
            pop_name = s[0] or 'all'
            input_name = '{}_spikes'.format(s[0] or 'input')
            spikes_file = os.path.abspath(s[1])

            spikes_ext = os.path.splitext(spikes_file)[1][1:]
            spikes_ext = 'sonata' if spikes_ext in ['h5', 'hdf5'] else spikes_ext

            inputs_dict[input_name] = {
                "input_type": "spikes",
                "module": spikes_ext,
                "input_file": spikes_file,
                "node_set": pop_name
            }
        if 'inputs' not in self._simulation_config:
            self._simulation_config['inputs'] = {}

        self._simulation_config['inputs'].update(inputs_dict)

    def _add_rates_inputs(self, rates_inputs):
        if rates_inputs is None:
            return

        inputs_dict = {}
        for s in rates_inputs:
            pop_name = s[0] or 'all'
            input_name = '{}_rates'.format(s[0] or 'input')
            rates_file = os.path.abspath(s[1])

            rates_ext = os.path.splitext(rates_file)[1][1:]
            rates_ext = 'sonata' if rates_ext in ['h5', 'hdf5'] else rates_ext

            inputs_dict[input_name] = {
                "input_type": rates_ext,
                "module": "pop_rates",
                "rates": rates_file,
                "node_set": pop_name
            }
        if 'inputs' not in self._simulation_config:
            self._simulation_config['inputs'] = {}

        self._simulation_config['inputs'].update(inputs_dict)

    def _add_run_params(self, tstart=0.0, tstop=1000.0, dt=0.001, **kwargs):
        self._simulation_config['run'] = {
            'tstart': tstart,
            'tstop': tstop,
            'dt': dt
        }

    def _copy_run_script(self, run_script=True, base_config='config.json'):
        if not run_script:
            return

        # Open the run_script, replace any variables ${...} with the appropiate value
        orig_run_script = 'run_{}.py'.format(self.bmtk_simulator)
        orig_path = os.path.join(self.examples_dir, orig_run_script)
        with open(orig_path, 'r') as fin:
            orig_script = fin.read()
            new_script = orig_script.replace('${CONFIG}', base_config)

        # Get the file name of the new run-script file. if "run_script" parameter is a string (eg run_bionet.test.py)
        #  then use that as the new script name. Otherwise if run_script==True then just use the old name.
        if isinstance(run_script, six.string_types):
            new_run_script = run_script
        elif isinstance(run_script, bool):
            new_run_script = orig_run_script
        else:
            logger.warning('Unable to create run-script "{}"'.format(run_script))
            return

        # Save the run script to
        new_path = os.path.join(self.base_dir, new_run_script)
        with open(new_path, 'w') as fout:
            fout.write(new_script)

    def _save_to_json(self, json_dict, config_file_name):
        logger.info('Creating config file: {}'.format(config_file_name))
        with open(os.path.join(self.base_dir, config_file_name), 'w') as outfile:
            ordered_dict = OrderedDict(sorted(json_dict.items(),
                                              key=lambda s: config_order.index(s[0]) if s[0] in config_order else 100))
            json.dump(ordered_dict, outfile, indent=2)

    def _save_config_single(self, config_file=None, config_name=None, overwrite=False):
        """Saves the simulation and circuit parameters to a single configuration file"""
        if config_file:
            # If users specified the --config-file option save the config into the file name specified by the user into
            # the given --base-dir, unless the config-file is referencing an absolute path.
            if os.path.isabs(config_file):
                cfg_path = config_file
            else:
                cfg_path = os.path.join(self._base_dir, config_file)

        elif config_name:
            # if users specified a configuration name, eg iclamp, then create config file config.iclamp.json. Take care
            #  if users confuse --config-name with --config-file. eg --config-name=config.iclamp.json, don't convert it
            #  to config.config.iclamp.json.json
            cfg_lower = config_name.lower()
            prefix = '' if cfg_lower.startswith('config') or cfg_lower.startswith('cfg') else 'config.'
            ext = '' if cfg_lower.endswith('json') else '.json'
            config_fname = '{}{}{}'.format(prefix, config_name, ext)
            config_fname = config_fname.replace(' ', '_') # in case user includes spaces in --config-name
            cfg_path = os.path.join(self._base_dir, config_fname)

        else:
            # If neither config_file or config_name just use ./config.json as file name
            cfg_path = os.path.abspath('config.{}.json'.format(self.bmtk_simulator.lower()))

        if os.path.exists(cfg_path) and not overwrite:
            # Throw error message if file exists and overwrite option is False
            logging.error('Configuration file {} already exists, skipping. '.format(cfg_path) +
                          'Please delete existing file, use a different name, or use overwrite=True.')
            return 'config.json'

        config_dir = os.path.dirname(cfg_path)
        if not os.path.exists(config_dir):
            try:
                os.makedirs(config_dir)
            except FileNotFoundError as fne:
                pass

        # Merge and the simulation and circuit configuration dictionaries - make sure to combine the "manifest"
        # section since it's the only section that will be in both configs.
        master_manifest = {}
        master_manifest.update(self._simulation_config.get('manifest', {}))
        master_manifest.update(self._circuit_config.get('manifest', {}))
        master_config = {}
        master_config.update(self._simulation_config)
        master_config.update(self._circuit_config)
        master_config['manifest'] = master_manifest
        self._save_to_json(master_config, cfg_path)

        return cfg_path

    def _save_config_split(self, config_file=None, config_name=None, overwrite=False):
        """Saves into multiple configuration files, a simulation config, a circuit config, and a master config that
        just contains reference to the other two. This is technically more correct in terms with the SONATA
        standard."""

        # Try to find a base-name to save all three configs under, eg. config.*.json, config.simulation_*.json and
        # config.circuit_*.json. Ideally the users will have specified in the --config-name option, but if not used
        # then check the --config-file option to parse out the base name
        if config_file:
            cfg_base_path = config_file
            cfg_name = cfg_base_path.replace('.json', '')
            cfg_sim_path = '{}_simulation.json'.format(cfg_name)
            cfg_circuit_path = '{}_circuit.json'.format(cfg_name)

        elif config_name:
            cfg_lower = config_name.lower()
            prefix = '' if cfg_lower.startswith('config') or cfg_lower.startswith('cfg') else 'config.'
            if cfg_lower.endswith('.json'):
                config_name, ext = os.path.splitext(config_name)
            else:
                ext = '.json'

            cfg_base_path = '{}{}{}'.format(prefix, config_name, ext)
            cfg_sim_path = '{}{}_simulation{}'.format(prefix, config_name, ext)
            cfg_circuit_path = '{}{}_circuit{}'.format(prefix, config_name, ext)

        else:
            # If neither config_file or config_name just use ./config.json as file name
            cfg_base_path = os.path.abspath('config.{}.json'.format(self.bmtk_simulator.lower()))
            cfg_sim_path = os.path.abspath('config.simulation.json')
            cfg_circuit_path = os.path.abspath('config.circuit.json')

        # Create directory with configs if it doesn't exists
        config_dir = os.path.dirname(cfg_base_path)
        if not os.path.exists(config_dir):
            try:
                os.makedirs(config_dir)
            except FileNotFoundError as fne:
                pass

        # Write and save base/combined config
        if os.path.exists(cfg_base_path) and not overwrite:
            logging.warning('Configuration file {} already exists, skipping.'.format(cfg_base_path) +
                            ' Please delete existing file, use a different name, or use overwrite=True.')
        else:
            base_config = {
                'network': cfg_circuit_path,
                'simulation': cfg_sim_path
            }
            self._save_to_json(base_config, cfg_base_path)

        # Write and save circuit config
        if os.path.exists(cfg_circuit_path) and not overwrite:
            logging.warning('Configuration file {} already exists, skipping.'.format(cfg_circuit_path) +
                            ' Please delete existing file, use a different name, or use overwrite=True.')
        else:
            self._save_to_json(self._circuit_config, cfg_circuit_path)

        # Write and save simulation config
        if os.path.exists(cfg_sim_path) and not overwrite:
            logging.warning('Configuration file {} already exists, skipping.'.format(cfg_sim_path) +
                            ' Please delete existing file, use a different name, or use overwrite=True.')
        else:
            self._save_to_json(self._simulation_config, cfg_sim_path)

        return cfg_base_path

    def build(self, include_examples=False, use_relative_paths=True, report_vars=[],
              report_nodes=None, clamp_reports=[], current_clamp=None, file_current_clamp=None,
              se_voltage_clamp=None,
              network_filter=None,
              spikes_inputs=None, rates_inputs=None, config_file='config.json', overwrite_config=False,
              config_name='', split_configs=False, run_script=True,
              **run_args):

        self._parse_network_dir(self.network_dir, network_filter=network_filter)
        self._create_components_dir(self.components_dir, with_examples=include_examples)
        if use_relative_paths:
            self._add_manifest(self._circuit_config, network_dir=self.network_dir, components_dir=self.components_dir)

        # selected_ns = self._create_node_sets_file(report_nodes)
        self._add_reports(report_vars, "all")
        self._add_clamp_reports(clamp_reports)
        self._add_output_section()
        self._simulation_config['target_simulator'] = self.target_simulator
        self._add_run_params(**run_args)
        self._add_current_clamp(current_clamp)
        self._add_file_current_clamp(file_current_clamp)
        self._add_se_voltage_clamp(se_voltage_clamp)
        self._add_spikes_inputs(spikes_inputs)
        self._add_rates_inputs(rates_inputs)

        if use_relative_paths:
            self._add_manifest(self._simulation_config, output_dir=self.output_dir)

        # Write the configurations to disk, either as a single file or split among base, circuit and simulation.
        if split_configs:
            cfg_path = self._save_config_split(
                config_file=config_file, config_name=config_name, overwrite=overwrite_config
            )
        else:
            cfg_path = self._save_config_single(
                config_file=config_file, config_name=config_name, overwrite=overwrite_config
            )

        if run_script:
            # Copy the run_{bionet|pointnet|filternet|popnet}.py script to the base-dir.
            self._copy_run_script(run_script=run_script, base_config=cfg_path)

    def compile_mechanisms(self):
        pass


class BioNetEnvBuilder(EnvBuilder):
    @property
    def examples_dir(self):
        return os.path.join(self.scripts_root, 'bionet')

    @property
    def target_simulator(self):
        return 'NEURON'

    @property
    def bmtk_simulator(self):
        return 'bionet'

    def _add_run_params(self, tstart=0.0, tstop=1000.0, dt=0.001, dL=20.0, spikes_threshold=-15.0, nsteps_block=5000,
                        v_init=-80.0, celsius=34.0, **kwargs):
        self._simulation_config['run'] = {
            'tstart': tstart,
            'tstop': tstop,
            'dt': dt,
            'dL': dL,
            'spike_threshold': spikes_threshold,
            'nsteps_block': nsteps_block
        }

        self._simulation_config['conditions'] = {
            'celsius': celsius,
            'v_init': v_init
        }

    def compile_mechanisms(self):
        mechanisms_dir = os.path.join(self.components_dir, 'mechanisms')
        logger.info('Attempting to compile NEURON mechanims under "{}"'.format(mechanisms_dir))

        modfiles_dir = os.path.join(mechanisms_dir, 'modfiles')
        if not os.path.exists(modfiles_dir):
            logger.warning('Could not find NEURON modfiles, attempting to copy over Allen Cell-Type modfiles.')
            copy_modfiles(mechanisms_dir=mechanisms_dir)

        compile_mechanisms(mechanisms_dir=mechanisms_dir)


class PointNetEnvBuilder(EnvBuilder):
    @property
    def examples_dir(self):
        return os.path.join(self.scripts_root, 'pointnet')

    @property
    def target_simulator(self):
        return 'NEST'

    @property
    def bmtk_simulator(self):
        return 'pointnet'

    def _add_output_section(self):
        super(PointNetEnvBuilder, self)._add_output_section()
        self._simulation_config['output']['quiet_simulator'] = True


class FilterNetEnvBuilder(EnvBuilder):
    @property
    def examples_dir(self):
        return os.path.join(self.scripts_root, 'filternet')

    @property
    def target_simulator(self):
        return 'LGNModel'

    @property
    def bmtk_simulator(self):
        return 'filternet'

    def _add_output_section(self):
        super(FilterNetEnvBuilder, self)._add_output_section()
        # for now not autmoatically includes rates.csv file since it is pretty big.
        # self._simulation_config['output']['rates_file_csv'] = 'rates.csv'
        self._simulation_config['output']['rates_h5'] = 'rates.h5'
        self._simulation_config['output']['spikes_file_csv'] = 'spikes.csv'
        # self._simulation_config['output']['spikes_file_h5'] = 'spikes.h5'


class PopNetEnvBuilder(EnvBuilder):
    @property
    def examples_dir(self):
        return os.path.join(self.scripts_root, 'popnet')

    @property
    def target_simulator(self):
        return 'DiPDE'

    @property
    def bmtk_simulator(self):
        return 'popnet'

    def _add_output_section(self):
        super(PopNetEnvBuilder, self)._add_output_section()
        if 'spikes_file' in self._simulation_config['output']:
            del self._simulation_config['output']['spikes_file']
        self._simulation_config['output']['rates_file'] = "firing_rates.csv"
