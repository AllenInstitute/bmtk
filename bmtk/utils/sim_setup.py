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
import six
from subprocess import call
from optparse import OptionParser
from collections import OrderedDict
import logging
from distutils.dir_util import copy_tree


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

network_dir_synonyms = ['network', 'networks', 'circuit', 'circuits', 'network_dir',  'circuit_dir']


class EnvBuilder(object):
    def __init__(self, base_dir='.', network_dir=None, components_dir=None, output_dir=None, node_sets_file=None):
        self._base_dir = self._get_base_dir(base_dir)
        self._network_dir = self._get_network_dir(network_dir)
        self._components_dir = self._get_components_dir(components_dir)
        self._output_dir = self._get_output_dir(output_dir)
        self._node_sets_file = self._get_node_sets_fname(node_sets_file)

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
        return os.path.join(local_path, 'scripts')

    @property
    def examples_dir(self):
        raise NotImplementedError()

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

    @property
    def network_dir(self):
        return self._network_dir

    def _get_network_dir(self, network_dir=None):
        """Attempts to get the appropiate path the the directory containing the sonata network files, either as specified
        by the user (-n <network_dir> argument) or else attempt to find the location of the network files.

        :param base_dir:
        :param network_dir:
        :return: The absolute path of the directory that does/should contain the network files
        """
        network_dir_abs = None

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

    def _parse_network_dir(self, network_dir):
        logger.info('Parsing {} for SONATA network files'.format(network_dir))
        net_nodes = {}
        net_edges = {}
        net_gaps = {}
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

    @property
    def components_dir(self):
        return self._components_dir

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

    @property
    def output_dir(self):
        return self._output_dir

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
        config_dict['manifest'] = {'$BASE_DIR': '${configdir}'}
        base_dir = os.path.abspath(self.base_dir)

        replace_str = lambda fd, bd, var_name: fd.replace(bd, var_name) if fd.startswith(bd) else fd

        if network_dir is not None:
            config_dict['manifest']['$NETWORK_DIR'] = replace_str(network_dir, base_dir, '$BASE_DIR')
            if len(config_dict['networks'].get('nodes', [])) > 0:
                config_dict['networks']['nodes'] = [{k: replace_str(v, network_dir, '$NETWORK_DIR')
                                                     for k, v in l.items()} for l in config_dict['networks']['nodes']]

            if len(config_dict['networks'].get('edges', [])) > 0:
                config_dict['networks']['edges'] = [{k: replace_str(v, network_dir, '$NETWORK_DIR')
                                                     for k, v in l.items()} for l in config_dict['networks']['edges']]

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

        if 'node_sets_file' in config_dict:
            config_dict['node_sets_file'] = replace_str(config_dict['node_sets_file'], base_dir, '$BASE_DIR')


    def _save_config(self, json_dict, config_file_name):
        logger.info('Creating config file: {}'.format(config_file_name))
        with open(os.path.join(self.base_dir, config_file_name), 'w') as outfile:
            ordered_dict = OrderedDict(sorted(json_dict.items(),
                                              key=lambda s: config_order.index(s[0]) if s[0] in config_order else 100))
            json.dump(ordered_dict, outfile, indent=2)

    @property
    def node_sets_file(self):
        return self._node_sets_file

    def _get_node_sets_fname(self, node_sets_file):
        if node_sets_file is None or not os.path.isabs(node_sets_file):
            abs_path = os.path.abspath(os.path.join(self.base_dir, node_sets_file or 'node_sets.json'))
        else:
            abs_path = node_sets_file

        return abs_path

    def _create_node_sets_file(self, recorded_nodes=None, default_ns='all'):
        if os.path.exists(self.node_sets_file):
            logger.info('Found existing node sets file: {}'.format(self.node_sets_file))
        else:
            logger.info('Creating new node sets file: {}'.format(self.node_sets_file))
            node_sets = {
                'biophysical_nodes': {'model_type': 'biophysical'},
                'point_nodes': {'model_type': 'point_process'}
            }
            if recorded_nodes is not None:
                node_sets['recorded_nodes'] = {'node_ids': recorded_nodes} if isinstance(recorded_nodes, list) \
                    else {'population': recorded_nodes}

            json.dump(node_sets, open(self.node_sets_file, 'w'), indent=2)

        if recorded_nodes is not None:
            default_ns = 'recorded_nodes'

        self._simulation_config['node_sets_file'] = self.node_sets_file
        return default_ns

    def _add_reports(self, cell_vars, node_set, section='soma'):
        if isinstance(cell_vars, six.string_types):
            cell_vars = [cell_vars]

        for v in cell_vars:
            logger.info('Adding membrane report for variable {}'.format(v))

        report_config = {
            '{}_report'.format(v): {
                'variable_name': v,
                'cells': node_set,
                'module': 'membrane_report',
                'sections': section
            } for v in cell_vars}

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
            "node_set": "all",
            "gids": current_param['gids'],
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

    def _add_run_params(self, tstart=0.0, tstop=1000.0, dt=0.001, **kwargs):
        self._simulation_config['run'] = {
            'tstart': tstart,
            'tstop': tstop,
            'dt': dt
        }

    def _copy_run_script(self):
        run_script = 'run_{}.py'.format(self.bmtk_simulator)
        shutil.copy(os.path.join(self.examples_dir, run_script), os.path.join(self.base_dir, run_script))

    def build(self, include_examples=False, use_relative_paths=True, report_vars=[],
              report_nodes=None, clamp_reports=[], current_clamp=None, file_current_clamp=None,
              se_voltage_clamp=None,
              spikes_inputs=None, config_file='config.json', **run_args):

        config_path = config_file if os.path.isabs(config_file) else os.path.join(self._base_dir, config_file)
        if os.path.exists(config_path):
            logger.info('Configuration file {} already exists, skipping.'.format(config_path))
        else:
            base_config = {
                'network': os.path.join(self._base_dir, 'circuit_config.json'),
                'simulation': os.path.join(self._base_dir, 'simulation_config.json')
            }
            self._save_config(base_config, config_path)

        self._parse_network_dir(self.network_dir)
        self._create_components_dir(self.components_dir, with_examples=include_examples)
        if use_relative_paths:
            self._add_manifest(self._circuit_config, network_dir=self.network_dir, components_dir=self.components_dir)
        self._save_config(self._circuit_config, 'circuit_config.json')

        selected_ns = self._create_node_sets_file(report_nodes)
        self._add_reports(report_vars, selected_ns)
        self._add_clamp_reports(clamp_reports)
        self._add_output_section()
        self._simulation_config['target_simulator'] = self.target_simulator
        # self._simulation_config['network'] = os.path.join(self.base_dir, 'circuit_config.json')
        self._add_run_params(**run_args)

        if current_clamp is not None:
            try:
                current_clamp['gids']
            except:
                current_clamp['gids']='all'

        self._add_current_clamp(current_clamp)

        if file_current_clamp is not None:
            self._add_file_current_clamp(file_current_clamp)

        if se_voltage_clamp is not None:
            try:
                se_voltage_clamp['gids']
            except:
                se_voltage_clamp['gids'] = 'all'

            self._add_se_voltage_clamp(se_voltage_clamp)

        if spikes_inputs!=None:
            self._add_spikes_inputs(spikes_inputs)
        if use_relative_paths:
            self._add_manifest(self._simulation_config, output_dir=self.output_dir)
        self._save_config(self._simulation_config, 'simulation_config.json')

        self._copy_run_script()


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
        cwd = os.getcwd()

        try:
            os.chdir(os.path.join(mechanisms_dir))
            call(['nrnivmodl', 'modfiles'])
            logger.info('  Success.')
        except Exception as e:
            logger.error('  Was unable to compile mechanism in {}'.format(mechanisms_dir))
        os.chdir(cwd)


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
        self._simulation_config['output']['rates_file_csv'] = 'rates.csv'
        self._simulation_config['output']['spikes_file_csv'] = 'spikes.csv'
        self._simulation_config['output']['spikes_file_h5'] = 'spikes.h5'


def build_env_bionet(base_dir='.', network_dir=None, components_dir=None, node_sets_file=None, include_examples=False,
                     tstart=0.0, tstop=1000.0, dt=0.001, dL=20.0, spikes_threshold=-15.0, nsteps_block=5000,
                     v_init=-80.0, celsius=34.0,
                     report_vars=[], report_nodes=None,
                     clamp_reports=[],
                     current_clamp=None,
                     file_current_clamp=None,
                     se_voltage_clamp=None,
                     spikes_inputs=None,
                     compile_mechanisms=False,
                     use_relative_paths=True,
                     config_file=None):
    env_builder = BioNetEnvBuilder(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
                                   node_sets_file=node_sets_file)

    env_builder.build(include_examples=include_examples, use_relative_paths=use_relative_paths,
                      report_vars=report_vars, report_nodes=report_nodes, clamp_reports=clamp_reports,
                       current_clamp=current_clamp,
                      file_current_clamp=file_current_clamp, se_voltage_clamp=se_voltage_clamp, spikes_inputs=spikes_inputs,
                      tstart=tstart, tstop=tstop, dt=dt, dL=dL, spikes_threshold=spikes_threshold,
                      nsteps_block=nsteps_block, v_init=v_init, celsius=celsius, config_file=config_file)

    if compile_mechanisms:
        env_builder.compile_mechanisms()


def build_env_pointnet(base_dir='.', network_dir=None, components_dir=None, node_sets_file=None, include_examples=False,
                       tstart=0.0, tstop=1000.0, dt=0.001, dL=20.0, spikes_threshold=-15.0, nsteps_block=5000,
                       v_init=-80.0, celsius=34.0,
                       report_vars=[], report_nodes=None, current_clamp=None,
                       spikes_inputs=None,
                       use_relative_paths=True,
                       config_file=None):
    env_builder = PointNetEnvBuilder(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
                                     node_sets_file=node_sets_file)

    env_builder.build(include_examples=include_examples, use_relative_paths=use_relative_paths,
                      report_vars=report_vars, report_nodes=report_nodes, current_clamp=current_clamp,
                      spikes_inputs=spikes_inputs,
                      tstart=tstart, tstop=tstop, dt=dt, dL=dL, spikes_threshold=spikes_threshold,
                      nsteps_block=nsteps_block, v_init=v_init, celsius=celsius)


def build_env_filternet(base_dir='.', network_dir=None, components_dir=None,
                        node_sets_file=None, include_examples=False, tstart=0.0, tstop=1000.0, config_file=None):
    env_builder = FilterNetEnvBuilder(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
                                     node_sets_file=node_sets_file)

    env_builder.build(include_examples=include_examples,
                      base_dir=base_dir, network_dir=network_dir, components_dir=components_dir, tstart=tstart,
                      tstop=tstop, config_file=config_file)



def build_env_popnet(base_dir='.', network_dir=None, reports=None, with_examples=True, tstop=1000.0, dt=0.001, **args):
    raise NotImplementedError()
    # simulator='popnet'
    # target_simulator='DiPDE'
    # components_dir='pop_components'
    #
    # # Copy run script
    # copy_run_script(base_dir=base_dir, simulator=simulator, run_script='run_{}.py'.format(simulator))
    #
    # # Build circuit_config and componenets directory
    # circuit_config = build_circuit_env(base_dir=base_dir, network_dir=network_dir, components_dir=components_dir,
    #                                    simulator=simulator, with_examples=with_examples)
    # circuit_config['components']['population_models_dir'] = '$COMPONENTS_DIR/population_models'
    # # population_models_dir = os.path.join(base_dir, components_dir, 'population_models')
    # if with_examples:
    #     models_dir =  os.path.join(base_dir, components_dir, 'population_models')
    #     if os.path.exists(models_dir):
    #         shutil.rmtree(models_dir)
    #     shutil.copytree(os.path.join(scripts_path, simulator, 'population_models'), models_dir)
    #
    # copy_config(base_dir, circuit_config, 'circuit_config.json')
    #
    # # Build simulation config
    # simulation_config = build_simulation_env(base_dir=base_dir, target_simulator=target_simulator, tstop=tstop, dt=dt,
    #                                          reports=reports)
    # # PopNet doesn't produce spike files so instead need to replace them with rates files
    # for output_key in simulation_config['output'].keys():
    #     if output_key.startswith('spikes'):
    #         del simulation_config['output'][output_key]
    # # simulation_config['output']['rates_file_csv'] = 'firing_rates.csv'
    # simulation_config['output']['rates_file'] = 'firing_rates.csv'
    #
    # copy_config(base_dir, simulation_config, 'simulation_config.json')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(module)s [%(levelname)s] %(message)s')

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
    parser.add_option('--include-examples', dest='include_examples', action='store_true', default=False,
                      help='Copies component files used by examples and tutorials.')
    parser.add_option('--compile-mechanisms', dest='compile_mechanisms', action='store_true', default=False,
                      help='Will try to compile the NEURON mechanisms (BioNet only).')
    parser.add_option('--config', dest='config_file', type='string', default=None,
                      help='Name of conguration json file.')

    options, args = parser.parse_args()

    # Check the passed in argments are correct. [sim] </path/to/dir/>
    if len(args) < 2:
        parser.error('Invalid number of arguments, Please specify a target simulation (bionet, pointnet, filternet,'
                     'popnet) and the path to save the simulation environment.')
    elif len(args) > 2:
        parser.error('Unrecognized arguments {}'.format(args[2:]))
    else:
        target_sim = args[0].lower()
        if target_sim not in ['bionet', 'popnet', 'pointnet', 'filternet']:
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
                         compile_mechanisms=options.compile_mechanisms,
                         config_file=options.config_file
                         )

    if target_sim == 'pointnet':
        build_env_pointnet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                         dt=options.dt, report_vars=options.mem_rep_vars, report_nodes=options.mem_rep_cells,
                         current_clamp=iclamp_args, include_examples=options.include_examples,
                         spikes_inputs=spikes_inputs)

    elif target_sim == 'popnet':
        build_env_popnet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                           dt=options.dt, config_file=options.config_file)

    elif target_sim == 'filternet':
        build_env_filternet(base_dir=base_dir, network_dir=options.network_dir, tstop=options.tstop,
                            include_examples=options.include_examples,
                            config_file=options.config_file)