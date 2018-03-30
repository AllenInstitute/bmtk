
class SimInput(object):
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
            del params['enabled']
        else:
            self.enabled = True

        # Fill in missing values with default (as specified by the subclass)
        for var_name, default_val in self._get_defaults():
            if var_name not in self.params:
                self.params[var_name] = default_val

        # Check there are no missing parameters

    def _get_defaults(self):
        return []

    @classmethod
    def build(cls, input_name, params):
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
    inputs_list = []
    for input_name, input_params in cfg.inputs.items():
        input_setting = SimInput.build(input_name, input_params)
        if input_setting.enabled:
            inputs_list.append(input_setting)


    #print inputs_list

    #exit()
    #print cfg.inputs
    #return None
    return inputs_list



