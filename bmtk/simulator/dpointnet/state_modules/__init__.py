import pandas as pd

from .zero_state import ZeroStateModule
from .randomized_state import RandomizedStateModule
from .input_state import InitStateFromInputModule
from .cached_states import CachedInitState

class StateModules:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.state_modules = {}
        self._initialized = True

    def add_init_state_module(self, module_cls, module_name=None, overwrite=True):
        if module_name is None or len(module_name) == 0:
            module_name = module_cls.module_name() 

        if module_name in self.state_modules and not overwrite:
            return
        else:
            self.state_modules[module_name] = module_cls

    def get_init_state_module(self, module_name):
        if module_name not in self.state_modules:
            raise ValueError(f'Could not find initial-state module "{module_name}".')
        
        return self.state_modules[module_name]
    
    def to_dict(self):
        return dict(self.state_modules)

    def to_list(self):
        ret_list = []
        for mod_name, mod_cls in self.state_modules.items():
            ret_list.append((mod_name, mod_cls.__module__))

        ret_list.sort(key=lambda i: i[0])
        return ret_list
    
    def to_dataframe(self):
        return pd.DataFrame(self.to_list(), columns=['module_name', 'module_class'])


StateModules().add_init_state_module(module_name='zero_state', module_cls=ZeroStateModule, overwrite=False)
StateModules().add_init_state_module(module_name='random_state', module_cls=RandomizedStateModule, overwrite=False)
StateModules().add_init_state_module(module_name='from_input', module_cls=InitStateFromInputModule, overwrite=False)
StateModules().add_init_state_module(module_name='cached_states', module_cls=CachedInitState, overwrite=False)