import logging
import re
from argparse import ArgumentParser

from bmtk.simulator.core.simulation_config import SimulationConfig


def inspect_bionet(config_path, format='json', output_path=None, filter=None):
    from bmtk.simulator import bionet

    conf = bionet.Config.from_json(config_path)
    conf.output['log_to_console'] = False
    conf.build_env()

    network = bionet.BioNetwork.from_config(conf)
    network.build_nodes()
    network.inspect_cells(
        format=format,
        output_path=output_path,
        filter=filter
    )

    bionet.nrn.quit_execution()


def inspect_pointnet(config_path, format='json', output_path=None, filter=None):
    from bmtk.simulator import pointnet

    conf = pointnet.Config.from_json(config_path)
    conf.output['log_to_console'] = False
    conf.build_env()

    network = pointnet.PointNetwork.from_config(conf)
    pointnet.PointSimulator.from_config(conf, network)
    network.inspect(
        format=format,
        output_path=output_path,
        filter=filter
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--to-json', action='store_true')
    parser.add_argument('--to-csv', action='store_true')
    parser.add_argument('--output-path', type=str)
    parser.add_argument('-p', '--population', type=str, required=False)
    parser.add_argument('--node-ids', type=str, required=False)
    parser.add_argument('config', type=str)
    args = parser.parse_args()

    config_path = args.config
    config_dict = SimulationConfig.load(config_path)
    target_sim = config_dict.target_simulator.upper()

    output_path = args.output_path
    format = 'json'
    if args.to_json and args.to_csv:
        raise ValueError('Both --to-csv and --to-json specified. Please specify a single output format!')
    elif args.to_csv:
        format = 'csv'

    filter_dict = {}
    if args.population:
        filter_dict['population'] = re.split(r'[,;\s]+', args.population)

    if args.node_ids:
        print(args.node_ids)
        node_ids = re.split(r'[,;\s]+', args.node_ids)
        node_ids = list(map(int, node_ids))
        filter_dict['node_id'] = node_ids

    if target_sim in ['BIONET', 'NEURON', 'NRN']:
        inspect_bionet(config_path=config_path, format=format, output_path=output_path, filter=filter_dict)
    elif target_sim in ['POINTNET', 'NEST']:
        inspect_pointnet(config_path=config_path, format=format, output_path=output_path, filter=filter_dict)
    elif target_sim in ['FILTERNET', 'LGN']:
        raise NotImplementedError
