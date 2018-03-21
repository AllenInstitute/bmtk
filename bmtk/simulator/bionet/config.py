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
import json

from neuron import h

#import bmtk.simulator.utils.config as msdk_config
from bmtk.utils.sonata.config import SonataConfig
from bmtk.simulator.utils.sim_validator import SimConfigValidator

from . import io
from . import nrn

pc = h.ParallelContext()    # object to access MPI methods
MPI_Rank = int(pc.id())


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


'''
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
'''


class Config(SonataConfig):
    @staticmethod
    def get_validator():
        return bionet_validator

    def create_output_dir(self):
        io.setup_output_dir(self.output_dir, self.log_file)

    def load_nrn_modules(self):
        nrn.load_neuron_modules(self.mechanisms_dir, self.templates_dir)
