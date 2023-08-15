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
import datetime
import re
import json
import warnings
from six import string_types


class SonataConfig(dict):
    """
    A special type of dict designed specially to handle SONATA configuration files.

    It handles things like manifest variables and combining network and circuit configurations into a single dictionary,
    as well as some (non-SONATA) special features. This class should be able to:
    * Combine circuit_config and simualation config files into a single dictionary automatically
    * resovlve manifest variables
    * Allow for custom and pre-defined variables for use in config
    * Make sure resolves the output_dir location


    General Usage:

        You can read in a root sonata config json file (or just the circuit or simulation files individual) by passing
        the file path or a file-pointer, which will return a dictionary like object you can use to build and run your
        simulation::

            sim_params = SonataConfig.from_json('/path/to/sonata_config.json')
            ... = sim_params['run']['tstop']

        You can also pass in a dictionary and use the from_dict() factory method::

            sim_params = SonataConfig.from_dict({...})

    JSON Variables:

        You can add variables to your SONATA config file which can make reusing json files easiers. The standard way
        of doing this by using the 'manifest' section to set the variable name and value to be used in other parts of
        the json. If the json file looks like::

            {
                'manifest': {'$CIRCUIT_HOME': '/home/john/cortex_simulation/network'},
                'network': {
                    'nodes': '${CIRCUIT_HOME}/nodes.h5,
                    'node_types': '${$CIRCUIT_HOME}/nodes_types.csv'
                }
            }

        SonataConfig will resolve the $CIRCUIT_HOME variable when accessed::

            sim_params['network']['nodes'] == '/home/john/cortex_simulation/network/nodes.h5'
            sim_params['network']['node_types'] == '/home/john/cortex_simulation/network/nodes_types.h5'

    Users and Predefined variables:

        You can also create your own predefined variables::

            circuit_config = {
                'run': {
                    "dt": "${time_step}",
                    "overwrite_output_dir": "${overwrite}"
                }
                'target_simulator': "${simulator}"
            }
            cfg = SonataConfig.from_dict(circuit_config, time_step=0.001, overwrite=True, simlator='CoreNeuron')
            cfg['run']['dt'] == 0.001
            cfg['run']['overwrite_output_dir'] == True
            cfg['target_simulator'] == 'CoreNeuron'

        There are also a number of built in variables which you can add to your json

        * ${configdir} - location of the current json file
        * ${workingdir} - directory where the code/simulator is being ran
        * ${datetime} - date-time string in Y-M-d_H-M-S format

    Validating the json file:

        You can check if the schema is valid, making suring certain section variables that are required actually exists
        in the config and is of the right type::

            cfg = SonataConfig.from_json('config.json')
            try:
                is_valid = cfg.valid()
            catch Exception e:
                is_valid = False

        If a section (eg. 'run', 'components', 'output', etc) is missing then it will be skipped, meaning if valid()
        returns true it may still be missing parts of the config to run a complete simulation.

        You can also add your own schema validator, just as long as it's a class instance that has a validate(cfg)
        method. The easiest way to do this is by using jsonschema::

            from jsonschema import Draft4Validator
            validator = Drafter4Validator('/path/to/schema.json')

            cfg = Sonata.from_json('config.json')
            cfg.validator = validator
            cfg.validate()
    """

    def __init__(self, *args, **kwargs):
        super(SonataConfig, self).__init__(*args, **kwargs)
        self._validator = None
        self._set_class_props()

    @property
    def validator(self):
        if self._validator is None:
            from jsonschema import Draft4Validator

            json_schema = os.path.join(os.path.dirname(__file__), 'config_schema.json')
            with open(json_schema, 'r') as f:
                config_schema = json.load(f)
                self._validator = Draft4Validator(schema=config_schema)

        return self._validator

    @validator.setter
    def validator(self, v):
        assert(hasattr(v, 'validate'))
        self._validator = v

    def validate(self):
        self.validator.validate(self)
        return True

    @classmethod
    def from_json(cls, config_file, validator=None, **opts):
        return cls(ConfigParser(**opts).parse(config_file))

    @classmethod
    def from_dict(cls, config_dict, validator=None, **opts):
        return cls(ConfigParser(**opts).parse(config_dict))

    @classmethod
    def from_yaml(cls, config_file, validator=None, **opts):
        raise NotImplementedError()

    @classmethod
    def load(cls, config_file, validator=None, **opts):
        # Implement factory method that can resolve the format/type of input configuration.
        if isinstance(config_file, dict):
            return cls.from_dict(config_file, validator, **opts)
        elif isinstance(config_file, string_types):
            if config_file.endswith('yml') or config_file.endswith('yaml'):
                return cls.from_yaml(config_file, validator, **opts)
            else:
                return cls.from_json(config_file, validator, **opts)
        else:
            raise Exception('Unable to open {}'.format(config_file))

    def _set_class_props(self):
        # Properties common to Sonata that will be regularly read by the simulators.
        # TODO: Use a (lazy?) adaptor, later versions of SONATA may change these parameters around
        self.run = self.get('run', {})
        self.conditions = self.get('conditions', {})
        self.components = self.get('components', {})
        self.output = self.get('output', {})
        self.inputs = self.get('inputs', {})
        self.reports = self.get('reports', {})
        self.networks = self.get('networks', {})

        self.output_dir = self.output.get('output_dir', '.')
        self.log_file = self.output.get('log_file', None)
        self.overwrite_output = self.output.get('overwrite_output_dir', False)

        self.mechanisms_dir = self.components.get('mechanisms_dir', None)
        self.templates_dir = self.components.get('templates_dir', None)

        self.nodes = self.networks.get('nodes', [])
        self.edges = self.networks.get('edges', [])
        self.gap_juncs = self.networks.get('gap_juncs', [])
        self.with_networks = 'networks' in self and len(self.nodes) > 0
        self.spike_threshold = self.run.get('spike_threshold', -15.0)
        self.dL = self.run.get('dL', 20.0)
        self.dt = self.run.get('dt', 0.1)
        self.tstart = self.run.get('tstart', 0.0)
        self.tstop = self.run.get('tstop', None)
        self.block_step = self.run.get('nsteps_block', 0)  # TODO: This is BioNet specific
        self.v_init = self.conditions.get('v_init', None)
        self.celsius = self.conditions.get('celsius', None)

        self.gid_mappings = self.get('gid_mapping_file', None)  # TODO: Remove
        self.node_sets = self.get('node_sets', {})


def copy_config(conf):
    """Copy configuration file to different directory, with manifest variables resolved.

    :param conf: configuration dictionary
    """
    output_dir = conf.output_dir
    config_name = os.path.basename(conf['config_path'])
    output_path = os.path.join(output_dir, config_name)
    with open(output_path, 'w') as fp:
        out_cfg = conf.copy()
        if 'manifest' in out_cfg:
            del out_cfg['manifest']
        json.dump(out_cfg, fp, indent=2)


def from_dict(config_dict, validator=None, **opts):
    """

    :param config_dict:
    :param validator:
    :param opts:
    :return:
    """
    warnings.warn('Deprecated: Please use SonataConfig.from_dict() instead.', DeprecationWarning)
    return SonataConfig.from_dict(config_dict, validator, **opts)


def from_json(config_file, validator=None, **opts):
    """Builds and validates a configuration json file.

    :param config_file: File object or path to a json file.
    :param validator: A SimConfigValidator object to validate json file. Won't validate if set to None
    :param opts:
    :return: A dictionary, verified against json validator and with manifest variables resolved.
    """
    warnings.warn('Deprecated: Please use SonataConfig.from_json() instead.', DeprecationWarning)
    return SonataConfig.from_json(config_file, validator, **opts)


def from_yaml(config_file, validator=None, **opts):
    """

    :param config_file:
    :param validator:
    :param opts:
    :return:
    """
    raise NotImplementedError()


class ConfigParser(object):
    """Helper class for parsing all the configs. Use the parse() method to return a normal dictionary to save into
    SonataConfig class
    """
    def __init__(self, **opts):
        self._usr_vars = opts  # dictionary of attributes passed in by the user, ie. not in MANIFEST

    @property
    def usr_vars(self):
        return self._usr_vars

    def parse(self, cfg_obj):
        """Builds and validates a configuration json dictionary object. Best to directly use from_json when possible.

        :param cfg_obj: Dictionary/json file containing a sonata configuration
        :return: A dictionary, verified against json validator and with manifest variables resolved.
        """
        if isinstance(cfg_obj, string_types):
            conf = json.load(open(cfg_obj, 'r'))
            conf['config_path'] = os.path.abspath(cfg_obj)
            conf['config_dir'] = os.path.dirname(conf['config_path'])
        elif isinstance(cfg_obj, dict):
            conf = cfg_obj.copy()
        else:
            raise Exception('{} is not a file or file path.'.format(cfg_obj))

        if 'config_path' not in conf:
            conf['config_path'] = os.path.join(os.getcwd(), 'sonata_config.json')
            conf['config_dir'] = os.path.dirname(conf['config_path'])

        # Build the manifest and resolve variables.
        manifest = self._build_manifest(conf)
        conf['manifest'] = manifest
        self._recursive_insert(conf, manifest)

        # In our work with Blue-Brain it was agreed that 'network' and 'simulator' parts of config may be split up into
        # separate files. If this is the case we build each sub-file separately and merge into this one
        for childconfig in ['network', 'simulation']:
            if childconfig in conf and isinstance(conf[childconfig], string_types):
                # Try to resolve the path of the network/simulation config files. If an absolute path isn't used find
                # the file relative to the current config file.
                conf_str = conf[childconfig]
                conf_path = conf_str if conf_str.startswith('/') else os.path.join(conf['config_dir'], conf_str)

                # Build individual json file and merge into parent.
                child_json = self.parse(conf_path)  # SonataConfig.from_json(conf_path)
                del child_json['config_path']  # we don't want 'config_path' of parent being overwritten.
                conf.update(child_json)

        self._post_process(conf, manifest)

        return conf

    def _special_variables(self, conf):
        """A list of preloaded variables to insert into the manifest, containing things like path to run-time directory,
        configuration directory, etc.
        """
        pre_manifest = dict()
        pre_manifest['workingdir'] = os.getcwd()
        if 'config_path' in conf:
            pre_manifest['configdir'] = os.path.dirname(conf['config_path'])  # path of configuration file
            pre_manifest['configfname'] = conf['config_path']

        dt_now = datetime.datetime.now()
        pre_manifest['time'] = dt_now.strftime('%H-%M-%S')
        pre_manifest['date'] = dt_now.strftime('%Y-%m-%d')
        pre_manifest['datetime'] = dt_now.strftime('%Y-%m-%d_%H-%M-%S')

        return pre_manifest

    def _build_manifest(self, conf):
        """Resolves the manifest section and resolve any internal variables"""
        if 'manifest' not in conf:
            return self._special_variables(conf)

        manifest = {}
        for key, val in conf['manifest'].items():
            nkey = key[1:] if key.startswith('$') else key
            manifest[nkey] = val

        resolved_manifest = self._special_variables(conf)
        resolved_keys = set()
        unresolved_keys = set(manifest.keys())

        # No longer using recursion since that can lead to an infinite loop if the person who writes the config file
        # isn't careful. Added code to allow for ${VAR} format in-case user wants to user "$.../some_${MODEl}_here/..."
        while unresolved_keys:
            for key in unresolved_keys:
                # Find all variables in manifest and see if they can be replaced by the value in resolved_manifest
                value = self._find_variables(manifest[key], resolved_manifest)

                # If value no longer references any variables and key-value pair to resolved_manifest and remove from
                # unresolved-keys
                if value.find('$') < 0:
                    resolved_manifest[key] = value
                    resolved_keys.add(key)

            # remove resolved key-value pairs from set, and make sure at every iteration unresolved_keys shrinks to
            # prevent infinite loops
            n_unresolved = len(unresolved_keys)
            unresolved_keys -= resolved_keys
            if n_unresolved == len(unresolved_keys):
                msg = "Unable to resolve manifest variables: {}".format(unresolved_keys)
                raise Exception(msg)

        return resolved_manifest

    def _recursive_insert(self, json_obj, manifest):
        """Loop through the config and substitute the path variables (e.g.: $MY_DIR) with the values from the manifest

        :param json_obj: A json dictionary object that may contain variables needing to be resolved.
        :param manifest: A dictionary of variable values
        :return: A new json dictionar config file with variables resolved
        """
        if isinstance(json_obj, string_types):
            return self._find_variables(json_obj, manifest)

        elif isinstance(json_obj, list):
            new_list = []
            for itm in json_obj:
                new_list.append(self._recursive_insert(itm, manifest))
            return new_list

        elif isinstance(json_obj, dict):
            for key, val in json_obj.items():
                if key == 'manifest':
                    continue
                json_obj[key] = self._recursive_insert(val, manifest)

            return json_obj

        else:
            return json_obj

    def _find_variables(self, json_str, manifest):
        """Replaces variables (i.e. $VAR, ${VAR}) with their values from the manifest.

        :param json_str: a json string that may contain none, one or multiple variable
        :param manifest: dictionary of variable lookup values
        :return: json rvalue with resolved variables. Won't resolve variables that don't exist in manifest.
        """
        ret_val = json_str
        variables = [m for m in re.finditer('\$\{?[\w]+\}?', json_str)]
        for var in variables:
            var_key = var.group()
            # change $VAR or ${VAR} --> VAR
            if var_key.startswith('${') and var_key.endswith('}'):
                # replace ${VAR} with VAR
                var_key = var_key[2:-1]
            elif var_key.startswith('$'):
                var_key = var_key[1:]

            # find variable value
            if var_key in self.usr_vars:
                rval = self.usr_vars[var_key]
            elif var_key in manifest:
                rval = manifest[var_key]
            else:
                continue

            if isinstance(rval, string_types) or len(json_str) > len(var.group()):
                ret_val = ret_val.replace(var.group(), str(rval))
            else:
                # In the case the variable value is not a string or not a part of the string - bool, float, etc. Try to
                # return the value directly
                return rval

        return ret_val

    def _post_process(self, conf, manifest):
        """Last parsing to do before """

        if 'output' in conf and 'output_dir' in conf['output']:
            # Special case for the 'output' section. Need to update certain pairs (log_file, spikes_file, etc) to
            # prepend the value of 'output_dir'. eg {'log_file': 'log.txt'} --> {'log_file': '${OUTPUT_DIR}/log.txt'}
            output_dir = conf['output']['output_dir']
            for k, file_path in conf['output'].items():
                if k == 'log_file' or k.startswith('spikes_file'):
                    if os.path.isabs(file_path) or file_path.startswith(output_dir):
                        # Skip if spikes/log file is an absolute path or already exists in a output_dir sub-dir
                        continue
                    else:
                        conf['output'][k] = os.path.join(output_dir, file_path)

        if 'node_sets' not in conf and 'node_sets_file' in conf:
            # Load in node_sets_file json if a reference to it exists
            conf['node_sets'] = json.load(open(conf['node_sets_file'], 'r'))
