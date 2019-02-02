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
import re
import copy
import datetime
from six import string_types


from bmtk.simulator.core.io_tools import io


class ConfigParser(object):
    def __init__(self, validator=None, **opts):
        self._validator = validator
        self._usr_vars = opts  # dictionary of attributes passed in by the user, ie. not in MANIFEST

    @property
    def validator(self):
        return self._validator

    @property
    def usr_vars(self):
        return self._usr_vars

    def parse(self, config_dict):
        """Builds and validates a configuration json dictionary object. Best to directly use from_json when possible.

        :param config_dict: Dictionary object
        :param validator: A SimConfigValidator object to validate json file. Won't validate if set to None
        :return: A dictionary, verified against json validator and with manifest variables resolved.
        """
        assert (isinstance(config_dict, dict))
        conf = copy.deepcopy(config_dict)  # Since the functions will mutate the dictionary we will copy just-in-case.

        if 'config_path' not in conf:
            conf['config_path'] = os.path.join(os.getcwd(), 'tmp_cfg.dict')
            conf['config_dir'] = os.path.dirname(conf['config_path'])

        # Build the manifest and resolve variables.
        # TODO: Check that manifest exists
        manifest = self._build_manifest(conf)
        conf['manifest'] = manifest
        self._recursive_insert(conf, manifest)

        # In our work with Blue-Brain it was agreed that 'network' and 'simulator' parts of config may be split up into
        # separate files. If this is the case we build each sub-file separately and merge into this one
        for childconfig in ['network', 'simulation']:
            if childconfig in conf and isinstance(conf[childconfig], string_types):
                # Try to resolve the path of the network/simulation config files. If an absolute path isn't used find
                # the file relative to the current config file. TODO: test if this will work on windows?
                conf_str = conf[childconfig]
                conf_path = conf_str if conf_str.startswith('/') else os.path.join(conf['config_dir'], conf_str)

                # Build individual json file and merge into parent.
                child_json = from_json(conf_path)
                del child_json['config_path']  # we don't want 'config_path' of parent being overwritten.
                conf.update(child_json)

        # Run the validator
        if self.validator is not None:
            self.validator.validate(conf)

        return conf

    def _special_variables(self, conf):
        """A list of preloaded variables to insert into the manifest, containing things like path to run-time directory,
        configuration directory, etc.
        """
        pre_manifest = dict()
        pre_manifest['workingdir'] = os.path.dirname(os.getcwd())
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

        # No longer using recursion since that can lead to an infinite loop if the person who writes the config file isn't
        # careful. Also added code to allow for ${VAR} format in-case user wants to user "$.../some_${MODEl}_here/..."
        while unresolved_keys:
            for key in unresolved_keys:
                # Find all variables in manifest and see if they can be replaced by the value in resolved_manifest
                value = self._find_variables(manifest[key], resolved_manifest)

                # If value no longer has variables, and key-value pair to resolved_manifest and remove from unresolved-keys
                if value.find('$') < 0:
                    resolved_manifest[key] = value
                    resolved_keys.add(key)

            # remove resolved key-value pairs from set, and make sure at every iteration unresolved_keys shrinks to prevent
            # infinite loops
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


'''
def from_json(config_file, validator=None, **opts):
    """Builds and validates a configuration json file.

    :param config_file: File object or path to a json file.
    :param validator: A SimConfigValidator object to validate json file. Won't validate if set to None
    :return: A dictionary, verified against json validator and with manifest variables resolved.
    """
    #print(config_file)
    #if os.path.isfile(config_file):
    #if isinstance(config_file, file):
    #    conf = json.load(config_file)
    if isinstance(config_file, string_types):
        conf = json.load(open(config_file, 'r'))
    elif isinstance(config_file, dict):
        conf = config_file.copy()
    else:
        raise Exception('{} is not a file or file path.'.format(config_file))

    # insert file path into dictionary
    if 'config_path' not in conf:
        conf['config_path'] = os.path.abspath(config_file)
        conf['config_dir'] = os.path.dirname(conf['config_path'])

    # Will resolve manifest variables and validate
    return from_dict(conf, validator, **opts)


def from_dict(config_dict, validator=None, **opts):
    """Builds and validates a configuration json dictionary object. Best to directly use from_json when possible.

    :param config_dict: Dictionary object
    :param validator: A SimConfigValidator object to validate json file. Won't validate if set to None
    :return: A dictionary, verified against json validator and with manifest variables resolved.
    """
    assert(isinstance(config_dict, dict))
    conf = copy.deepcopy(config_dict)  # Since the functions will mutate the dictionary we will copy just-in-case.

    if 'config_path' not in conf:
        conf['config_path'] = os.path.join(os.getcwd(), 'tmp_cfg.dict')
        conf['config_dir'] = os.path.dirname(conf['config_path'])

    # Build the manifest and resolve variables.
    # TODO: Check that manifest exists
    manifest = __build_manifest(conf, **opts)
    conf['manifest'] = manifest
    __recursive_insert(conf, manifest, **opts)

    # In our work with Blue-Brain it was agreed that 'network' and 'simulator' parts of config may be split up into
    # separate files. If this is the case we build each sub-file separately and merge into this one
    for childconfig in ['network', 'simulation']:
        if childconfig in conf and isinstance(conf[childconfig], string_types):
            # Try to resolve the path of the network/simulation config files. If an absolute path isn't used find
            # the file relative to the current config file. TODO: test if this will work on windows?
            conf_str = conf[childconfig]
            conf_path = conf_str if conf_str.startswith('/') else os.path.join(conf['config_dir'], conf_str)

            # Build individual json file and merge into parent.
            child_json = from_json(conf_path)
            del child_json['config_path']  # we don't want 'config_path' of parent being overwritten.
            conf.update(child_json)

    # Run the validator
    if validator is not None:
        validator.validate(conf)

    return conf


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


def __special_variables(conf):
    """A list of preloaded variables to insert into the manifest, containing things like path to run-time directory,
    configuration directory, etc.
    """
    pre_manifest = dict()
    pre_manifest['workingdir'] = os.path.dirname(os.getcwd())
    if 'config_path' in conf:
        pre_manifest['configdir'] = os.path.dirname(conf['config_path'])  # path of configuration file
        pre_manifest['configfname'] = conf['config_path']

    dt_now = datetime.datetime.now()
    pre_manifest['time'] = dt_now.strftime('%H-%M-%S')
    pre_manifest['date'] = dt_now.strftime('%Y-%m-%d')
    pre_manifest['datetime'] = dt_now.strftime('%Y-%m-%d_%H-%M-%S')

    return pre_manifest


def __build_manifest(conf, **opts):
    """Resolves the manifest section and resolve any internal variables"""
    if 'manifest' not in conf:
        return __special_variables(conf)

    manifest = {}
    for key, val in conf['manifest'].items():
        nkey = key[1:] if key.startswith('$') else key
        manifest[nkey] = val

    resolved_manifest = __special_variables(conf)
    resolved_keys = set()
    unresolved_keys = set(manifest.keys())

    # No longer using recursion since that can lead to an infinite loop if the person who writes the config file isn't
    # careful. Also added code to allow for ${VAR} format in-case user wants to user "$.../some_${MODEl}_here/..."
    while unresolved_keys:
        for key in unresolved_keys:
            # Find all variables in manifest and see if they can be replaced by the value in resolved_manifest
            value = __find_variables(manifest[key], resolved_manifest, **opts)

            # If value no longer has variables, and key-value pair to resolved_manifest and remove from unresolved-keys
            if value.find('$') < 0:
                resolved_manifest[key] = value
                resolved_keys.add(key)

        # remove resolved key-value pairs from set, and make sure at every iteration unresolved_keys shrinks to prevent
        # infinite loops
        n_unresolved = len(unresolved_keys)
        unresolved_keys -= resolved_keys
        if n_unresolved == len(unresolved_keys):
            msg = "Unable to resolve manifest variables: {}".format(unresolved_keys)
            raise Exception(msg)

    return resolved_manifest


def __recursive_insert(json_obj, manifest, **opts):
    """Loop through the config and substitute the path variables (e.g.: $MY_DIR) with the values from the manifest

    :param json_obj: A json dictionary object that may contain variables needing to be resolved.
    :param manifest: A dictionary of variable values
    :return: A new json dictionar config file with variables resolved
    """
    if isinstance(json_obj, string_types):
        return __find_variables(json_obj, manifest, **opts)

    elif isinstance(json_obj, list):
        new_list = []
        for itm in json_obj:
            new_list.append(__recursive_insert(itm, manifest, **opts))
        return new_list

    elif isinstance(json_obj, dict):
        for key, val in json_obj.items():
            if key == 'manifest':
                continue
            json_obj[key] = __recursive_insert(val, manifest, **opts)

        return json_obj

    else:
        return json_obj


def __find_variables(json_str, manifest, **opts):
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
        if var_key in opts:
            rval = opts[var_key]
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
'''

class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self._env_built = False
        self._io = None

        self._node_set = {}
        self._load_node_set()

    @property
    def io(self):
        if self._io is None:
            self._io = io
        return self._io

    @io.setter
    def io(self, io):
        self._io = io

    @property
    def run(self):
        return self.get('run', {})

    @property
    def tstart(self):
        return self.run.get('tstart', 0.0)

    @property
    def tstop(self):
        return self.run['tstop']

    @property
    def dt(self):
        return self.run.get('dt', 0.1)

    @property
    def spike_threshold(self):
        return self.run.get('spike_threshold', -15.0)

    @property
    def dL(self):
        return self.run.get('dL', 20.0)

    @property
    def gid_mappings(self):
        return self.get('gid_mapping_file', None)

    @property
    def block_step(self):
        return self.run.get('nsteps_block', 5000)

    @property
    def conditions(self):
        return self.get('conditions', {})

    @property
    def celsius(self):
        return self.conditions['celsius']

    @property
    def v_init(self):
        return self.conditions['v_init']

    @property
    def path(self):
        return self['config_path']

    @property
    def output(self):
        return self['output']

    @property
    def output_dir(self):
        return self.output['output_dir']

    @property
    def overwrite_output(self):
        return self.output.get('overwrite_output_dir', False)

    @property
    def log_file(self):
        return self.output['log_file']

    @property
    def components(self):
        return self.get('components', {})

    @property
    def morphologies_dir(self):
        return self.components['morphologies_dir']

    @property
    def synaptic_models_dir(self):
        return self.components['synaptic_models_dir']

    @property
    def point_neuron_models_dir(self):
        return self.components['point_neuron_models_dir']

    @property
    def mechanisms_dir(self):
        return self.components.get('mechanisms_dir', None)

    @property
    def biophysical_neuron_models_dir(self):
        return self.components['biophysical_neuron_models_dir']

    @property
    def templates_dir(self):
        return self.components.get('templates_dir', None)

    @property
    def with_networks(self):
        return 'networks' in self and len(self.nodes) > 0

    @property
    def networks(self):
        return self['networks']

    @property
    def nodes(self):
        return self.networks.get('nodes', [])

    @property
    def edges(self):
        return self.networks.get('edges', [])

    @property
    def reports(self):
        return self.get('reports', {})

    @property
    def inputs(self):
        return self.get('inputs', {})

    @property
    def node_sets(self):
        return self._node_set

    @property
    def spikes_file(self):
        return os.path.join(self.output_dir, self.output['spikes_file'])

    def _load_node_set(self):
        if 'node_sets_file' in self.keys():
            node_set_val = self['node_sets_file']
        elif 'node_sets' in self.keys():
            node_set_val = self['node_sets']
        else:
            self._node_set = {}
            return

        if isinstance(node_set_val, dict):
            self._node_set = node_set_val
        else:
            try:
                self._node_set = json.load(open(node_set_val, 'r'))
            except Exception as e:
                io.log_exception('Unable to load node_sets_file {}'.format(node_set_val))

    def copy_to_output(self):
        copy_config(self)

    def get_modules(self, module_name):
        return [report for report in self.reports.values() if report['module'] == module_name]

    def _set_logging(self):
        """Check if log-level and/or log-format string is being changed through the config"""
        output_sec = self.output
        if 'log_format' in output_sec:
            self._io.set_log_format(output_sec['log_format'])

        if 'log_level' in output_sec:
            self._io.set_log_level(output_sec['log_level'])

        if 'log_to_console' in output_sec:
            self._io.log_to_console = output_sec['log_to_console']

        if 'quiet_simulator' in output_sec and output_sec['quiet_simulator']:
            self._io.quiet_simulator()

    def build_env(self):
        if self._env_built:
            return

        self._set_logging()
        self.io.setup_output_dir(self.output_dir, self.log_file, self.overwrite_output)
        self.copy_to_output()
        self._env_built = True

    @staticmethod
    def get_validator():
        raise NotImplementedError

    @classmethod
    def from_json(cls, config_file, validate=False, **opts):
        validator = cls.get_validator() if validate else None
        return cls(from_json(config_file, validator, **opts))

    @classmethod
    def from_dict(cls, config_dict, validate=False):
        validator = cls.get_validator() if validate else None
        return cls(from_dict(config_dict, validator))

    @classmethod
    def from_yaml(cls, config_file, validate=False):
        raise NotImplementedError

    @classmethod
    def load(cls, config_file, validate=False):
        # Implement factory method that can resolve the format/type of input configuration.
        if isinstance(config_file, dict):
            return cls.from_dict(config_file, validate)
        elif isinstance(config_file, string_types):
            if config_file.endswith('yml') or config_file.endswith('yaml'):
                return cls.from_yaml(config_file, validate)
            else:
                return cls.from_json(config_file, validate)
        else:
            raise Exception


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
    return ConfigParser(validator=validator, **opts).parse(config_dict)


def from_json(config_file, validator=None, **opts):
    """Builds and validates a configuration json file.

    :param config_file: File object or path to a json file.
    :param validator: A SimConfigValidator object to validate json file. Won't validate if set to None
    :param opts:
    :return: A dictionary, verified against json validator and with manifest variables resolved.
    """
    if isinstance(config_file, string_types):
        conf = json.load(open(config_file, 'r'))
    elif isinstance(config_file, dict):
        conf = config_file.copy()
    else:
        raise Exception('{} is not a file or file path.'.format(config_file))

    # insert file path into dictionary
    if 'config_path' not in conf:
        conf['config_path'] = os.path.abspath(config_file)
        conf['config_dir'] = os.path.dirname(conf['config_path'])

    # Will resolve manifest variables and validate
    return ConfigParser(validator=validator, **opts).parse(conf)


def from_yaml(config_file, validator=None, **opts):
    """

    :param config_file:
    :param validator:
    :param opts:
    :return:
    """
    raise NotImplementedError()
