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


order_lookup = {k: i for i, k in enumerate(config_order)}
def sort_config_keys(ckey):
    print(ckey)
    exit()


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
    parser.add_option('-r', '--run-time', type='float', dest='run_time', default=0.0)

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
    parser.add_option('--membrane_report_file', dest='mem_rep_file', type='string', action='callback',
                      callback=membrane_report_parser, default='$OUTPUT_DIR/cell_vars.h5')
    parser.add_option('--membrane_report-sections', dest='mem_rep_secs', type='string', action='callback',
                      callback=membrane_report_parser, default='all')

    options, args = parser.parse_args()
    reports = {}

    if options.has_membrane_report:
        reports['membrane_report'] = {
            'module': 'membrane_report',
            'variable_name': options.mem_rep_vars,
            'cells': options.mem_rep_cells,
            'file_name': options.mem_rep_file,
            'sections': options.mem_rep_secs,
        }

    target_sim = args[0].lower() if len(args) == 1 else None
    if target_sim not in ['bionet', 'popnet', 'pointnet', 'mintnet']:
        raise Exception('Must specify one target simulator. options: "bionet", pointnet", "popnet" or "mintnet"')

    if target_sim == 'bionet':
        build_env_bionet(base_dir=options.base_dir, network_dir=options.network_dir, run_time=options.run_time,
                         reports=reports)
