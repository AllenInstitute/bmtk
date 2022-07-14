import logging
from optparse import OptionParser

from .create_environment import create_environment


logger = logging.getLogger(__name__)


def __list_parser(option, opt, value, parser):
    """Helper function for parsing inputs, since they may contain a list of values that are comma (,) separated.

    eg. ---spike-trains=lgn:inputs/lgn_spikes.h5,bkg:inputs/bkg_spikes.h5
    """
    # parser.values.has_membrane_report = True
    if ',' in value:
        try:
            setattr(parser.values, option.dest, [float(v) for v in value.split(',')])
        except ValueError as ve:
            setattr(parser.values, option.dest, value.split(','))

    else:
        setattr(parser.values, option.dest, value)


def __split_list(options, opt_name):
    opt_vals = getattr(options, opt_name)
    updated_vals = []
    if opt_vals is not None:
        opt_strings = [opt_vals] if isinstance(opt_vals, str) else list(opt_vals)
        for opt_str in opt_strings:
            vals = opt_str.split(':')
            if len(vals) == 1:
                updated_vals.append((None, vals[0]))
            elif len(vals) == 2:
                updated_vals.append((vals[0], vals[1]))
            else:
                parser.error('Cannot parse spike-input string <pop1>:<spikes-file1>,<pop2>:<spikes-file2>,...')

        setattr(options, opt_name, updated_vals)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO, format='%(module)s [%(levelname)s] %(message)s')
    logging.basicConfig(level=logging.WARN, format='bmtk.utils.create_environment [%(levelname)s] %(message)s')

    parser = OptionParser(usage="Usage: python %prog [options] [bionet|pointnet|popnet|filternet] sim_dir")
    parser.add_option('-n', '--network-dir', dest='network_dir', default=None,
                      help="Use an exsting directory with network files.")
    parser.add_option('-o', '--output-dir', dest='output_dir', default=None,
                      help="Directory to use for storing output of simulations.")
    parser.add_option('-c', '--components-dir', dest='components_dir', default=None,
                      help="Directory to use for parameter files, morphology, and other used components.")
    parser.add_option('--tstop', type='float', dest='tstop', default=1000.0)
    parser.add_option('--dt', type=float, dest='dt', help='simulation time step dt', default=0.001)
    parser.add_option('--report-vars', dest='report_vars', type='string', action='callback',
                      callback=__list_parser, default=[],
                      help='A list of membrane variables to record from; v, cai, etc.')
    parser.add_option('--report-nodes', dest='report_nodes', type='string', action='callback',
                      callback=__list_parser, default=None)
    parser.add_option('--iclamp', dest='current_clamp', type='string', action='callback',
                      callback=__list_parser, default=None,
                      help='Adds a soma current clamp using three variables: <amp>,<delay>,<duration> (nA, ms, ms)')
    parser.add_option('--spikes-inputs', dest='spikes_inputs', type='string', action='callback', callback=__list_parser,
                      default=None,
                      help='Spike trains to use for inputs stimulations, a list in the form of ' 
                           '<pop1>:<spikes_file_1.h5>[,<pop2>:<spikes_file_1.h5>,...]')
    parser.add_option('--rates-inputs', dest='rates_inputs', type='string', action='callback', callback=__list_parser,
                      default=None,
                      help='Population rates file to use for inputs stimulations, a list in the form of ' 
                           '<pop1>:<rates_file_1.h5>[,<pop2>:<ratess_file_1.h5>,...]')
    parser.add_option('--include-examples', dest='include_examples', action='store_true', default=False,
                      help='Copies component files used by examples and tutorials.')
    parser.add_option('--compile-mechanisms', dest='compile_mechanisms', action='store_true', default=False,
                      help='Will try to compile the NEURON mechanisms (BioNet only).')
    parser.add_option('--config-file', dest='config_file', type='string', default=None,
                      help='File name of conguration json file.')
    parser.add_option('--config-name', dest='config_name', type='string', default=None,
                      help='General name of environment configuration file(s).')
    parser.add_option('--split-configs', dest='split_configs', action='store_true', default=False,
                      help='If option is used then the configuration files will be split into different circuit and '
                      'simulation configuration files')
    parser.add_option('--network-filter', dest='network_filter', type='string', action='callback',
                      callback=__list_parser, default=None,
                      help='Option to filter which files from --network-dir are included in configuration, A list of '
                      'files, v1_node.h5,v1_node_types.csv or a string v1,lgn.')
    parser.add_option('--run-script', dest='run_script', type='string', default=None,
                      help='Name of bmtk simulation run *.py script used to run the simulation. By default name will '
                      'be run_{simulator}.py')
    parser.add_option('--overwrite', dest='overwrite', action='store_true', default=False,
                      help='Overwrite existing configuration/run scripts if exists.')

    options, args = parser.parse_args()

    # Check the passed in argments are correct. [sim] </path/to/dir/>
    if len(args) < 2:
        parser.error('Invalid number of arguments, Please specify a target simulation (bionet, pointnet, filternet,'
                     'popnet) and the path to save the simulation environment.')
        exit(1)
    elif len(args) > 2:
        parser.error('Unrecognized arguments {}'.format(args[2:]))
        exit(1)
    else:
        simulator = args[0].lower()
        if simulator not in ['bionet', 'popnet', 'pointnet', 'filternet']:
            parser.error('Must specify one target simulator. options: "bionet", pointnet", "popnet", "filternet"')
        base_dir = args[1]

    if options.current_clamp is not None:
        cc_args = options.current_clamp
        if len(cc_args) != 3:
            parser.error('Invalid arguments for current clamp, requires three floating point numbers '
                         '<ampage>,<delay>,<duration> (nA, ms, ms)')
        iclamp_args = {'amp': float(cc_args[0]), 'delay': float(cc_args[1]), 'duration': float(cc_args[2])}
        options.current_clamp = iclamp_args
    else:
        options.current_clamp = None

    __split_list(options, 'spikes_inputs')
    __split_list(options, 'report_vars')
    __split_list(options, 'rates_inputs')

    if options.run_script is None:
        options.run_script = True

    create_environment(simulator=simulator, base_dir=base_dir, **vars(options))
