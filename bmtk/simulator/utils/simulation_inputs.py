
class SimInput(object):
    """A helper class for parsing the "inputs" section of a SONATA config file. Separate the actual parameters needed
    to instantiate a module from the metadata parameters (eg, name, module, input_type)

    Use the build() method to parse the json/dictionary of a section of input, then **params** to get a dictionary
    of the values used to instantiate::

        input = SimInput.Build(
                  'spikes_inputs',
                  {'module': 'spikes', 'input_type': 'hdf5', input_file: 'my_spikes.h5', 'time_scale': 'ms'}
        )

        if input.module == 'spike':
            SpikesInput(**params)
            ...

    Attributes:

        * **name** - str, name of module
        * **input_type** - str
        * **module** - str,
        * **params** - dictionary, all parameters (not including name, input_type, module)

    Custom Modules:

        Sometimes certain input types may require extra steps in processing, like auto-conversion of filling in missing
        parameters. In this case use the register module method::

            class MyVClampInput(SimInput):
                def avail_module():
                    return ['vclamp', 'voltage_clamp']

                def build():
                    ....

            SimInput.registerModule(MyVClampInput)

        Then when SimInput.build() is called and the 'module_name'=='vclamp' (or 'voltage_clamp') it will pass the parsing
        to the MyVClampInput class.
    """
    registry = {}  # For factory function

    def __init__(self, input_name, input_type, module, params):
        self.name = input_name
        self.input_type = input_type
        self.module = module
        self.params = params.copy()

        # Remove the 'module' and 'input_type' from the params since user should access it through the variable
        for param_key in ['module', 'input_type']:
            if param_key in self.params:
                del self.params[param_key]

        # Special variable, not a part of standard but still want for ease of testing
        if 'enabled' in params:
            self.enabled = params['enabled']
            del self.params['enabled']
        else:
            self.enabled = True

        # Fill in missing values with default (as specified by the subclass)
        for var_name, default_val in self._get_defaults():
            if var_name not in self.params:
                self.params[var_name] = default_val

        # Check there are no missing parameters

    @property
    def node_set(self):
        return self.params.get('node_set', None)

    def _get_defaults(self):
        return []

    @classmethod
    def build(cls, input_name, params):
        """Creates a SimInput object with parsed out parameters

        :param input_name: name of specific input
        :param params: dictionary of input parameters
        :return: SimInput object, or subclass that matches the specified 'module' value
        """
        params = params.copy()
        if 'module' not in params:
            raise Exception('inputs setting {} does not specify the "module".'.format(input_name))

        if 'input_type' not in params:
            raise Exception('inputs setting {} does not specify the "input_type".'.format(input_name))

        module_name = params['module']
        input_type = params['input_type']
        module_cls = SimInput.registry.get(module_name, SimInput)

        return module_cls(input_name, input_type, module_name, params)

    @classmethod
    def register_module(cls, subclass):
        """

        :param subclass:
        """
        # For factory, register subclass based on the module name(s)
        assert(issubclass(subclass, cls))
        mod_registry = cls.registry
        mod_list = subclass.avail_modules()
        modules = mod_list if isinstance(mod_list, list) else [mod_list]
        for mod_name in modules:
            if mod_name in mod_registry:
                raise Exception('Multiple modules named {}'.format(mod_name))
            mod_registry[mod_name] = subclass

        return subclass


def from_config(cfg):
    """Takes in a bmtk.utils.Config instance and will automatically parse each "input" in the config sections,
    returning a list of SimInput objects. If an input has "enabled" = False then it will automatically be excluded.

    :param cfg: A SONATAConfig object
    :return: A list of SimInput modules corresponding to the parsed inputs of a config.
    """
    inputs_list = []
    for input_name, input_params in cfg.inputs.items():
        input_setting = SimInput.build(input_name, input_params)
        if input_setting.enabled:
            inputs_list.append(input_setting)

    return inputs_list



