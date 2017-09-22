import os
import json

import bmtk.simulator.utils.config as msdk_config
from bmtk.simulator.utils.sim_validator import SimConfigValidator

# load the configuration schema
schema_folder = os.path.join(os.path.dirname(__file__), 'schemas')
config_schema_file = os.path.join(schema_folder, 'config_schema.json')

# json schemas (but not real jsonschema) to describe the various input file formats
file_formats = [
    ("csv:nodes_internal", os.path.join(schema_folder, 'csv_nodes_internal.json')),
    ("csv:node_types_internal", os.path.join(schema_folder, 'csv_node_types_internal.json')),
    ("csv:edge_types", os.path.join(schema_folder, 'csv_edge_types.json')),
    ("csv:nodes_external", os.path.join(schema_folder, 'csv_nodes_external.json')),
    ("csv:node_types_external", os.path.join(schema_folder, 'csv_node_types_external.json'))
]

# Create a config and input file validator for Bionet
with open(config_schema_file, 'r') as f:
    config_schema = json.load(f)
bionet_validator = SimConfigValidator(config_schema, file_formats=file_formats)


def from_json(config_file, validate=True):
    """Converts a config file into a dictionary. Will resolve manifest variables, validate schema and input files, as
    well as other behind-the-scenes actions required by bionet.

    :param config_file: json file object or path to json configuration file
    :param validate: will validate the config file against schemas/config_schema.json (Default True)
    :return: config json file in dictionary format
    """
    validator = bionet_validator if validate else None
    return msdk_config.from_json(config_file, validator)


def from_dict(config_dict, validate=True):
    """Same as from_json, but allows for direct input of dictionary object (use from_json when possible).

    :param config_dict:
    :param validate:
    :return:
    """
    validator = bionet_validator if validate else None
    return msdk_config.from_dict(config_dict, validator)

def copy(config_file):
    return msdk_config.copy_config(config_file)