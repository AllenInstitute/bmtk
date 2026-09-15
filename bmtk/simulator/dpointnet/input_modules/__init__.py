from collections import defaultdict
import pandas as pd

from .inputs_base import InputsGeneratorMod
from .delayed_cue_spikes import DelayedCueSpikes
from .lgn_generator import LGNGenerator
from .noisy_current import NoisyCurrent, PoissonSpikesInternal
from .poisson_spikes import PoissonSpikes
from .rand_spikes_generator import BernoulliSpikes, RandomSpikesGenerator
from .spikes_files_generator import SpikesFilesGenerator
from .spikes_function_generator import SpikesFunctionGenerator, spikes_function


class InputModules:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._input_module_classes = defaultdict(dict)
        self._initialized = True

    def add_module(self, module_cls, input_type=None, module_name=None, overwrite=True):
        if input_type is None or len(input_type) == 0:
            input_type = module_cls.input_type()

        if module_name is None or len(module_name) == 0:
            module_name = module_cls.module()

        if module_name in self._input_module_classes[input_type] and not overwrite:
            return
        else:
            self._input_module_classes[input_type][module_name] = module_cls

    def get_module(self, input_type, module_name):
        if module_name not in self._input_module_classes[input_type]:
            raise ValueError(
                f'Could not find module "{module_name}" for input type "{input_type}"'
            )

        return self._input_module_classes[input_type][module_name]

    def to_dict(self):
        return dict(self._input_module_classes)

    def to_list(self):
        ret_list = []
        for in_type, mod_dict in self._input_module_classes.items():
            for mod_name, mod_cls in mod_dict.items():
                ret_list.append((in_type, mod_name, mod_cls.__module__))

        ret_list.sort(key=lambda i: i[0] + "_" + i[1])
        return ret_list

    def to_dataframe(self):
        return pd.DataFrame(
            self.to_list(), columns=["input_type", "module_name", "module_class"]
        )


# input_modules_lu = InputModules()
InputModules().add_module(NoisyCurrent, overwrite=False)
InputModules().add_module(PoissonSpikesInternal, overwrite=False)
InputModules().add_module(PoissonSpikes, overwrite=False)
InputModules().add_module(LGNGenerator, overwrite=False)
InputModules().add_module(LGNGenerator, module_name="lgn_tf", overwrite=False)
InputModules().add_module(RandomSpikesGenerator, overwrite=False)
InputModules().add_module(BernoulliSpikes, overwrite=False)
InputModules().add_module(SpikesFilesGenerator, overwrite=False)
InputModules().add_module(SpikesFunctionGenerator, overwrite=False)
InputModules().add_module(DelayedCueSpikes, overwrite=False)
