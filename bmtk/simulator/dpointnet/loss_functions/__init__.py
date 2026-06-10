from .spike_rate_distribution_target import SpikeRateDistributionTarget
from .target_firing_rates import TargetFiringRate
from .orientation_selectivity_loss import OrientationSelectivityLoss
from .voltage_regularization import VoltageRegularization
from .synchronization_loss import SynchronizationLoss
from .weight_regularization import EMDWeightRegularization


class LossModules:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._loss_modules = {}
        self._initialized = True

    def add_module(self, module_cls, module_name=None, overwrite=True):
        if module_name is None or len(module_name) == 0:
            module_name = module_cls.module()
        
        if module_name in self._loss_modules and not overwrite:
            return
        else:
            self._loss_modules[module_name] = module_cls

    def get_module(self, module_name):
        if module_name not in self._loss_modules:
            raise ValueError(f'{LossModules.__class__.__name__}: Could not find loss function module "{module_name}".')
        
        return self._loss_modules[module_name]


def add_loss_module(mod, module_name=None, overwrite=True):
    LossModules().add_module(mod, module_name=module_name, overwrite=overwrite)


def register_loss_module(_cls=None, *_, **wkargs):
    def decorator(cls):
        mod_name = wkargs.get('module_name', None)
        overwrite = wkargs.get('overwrite', True)
        LossModules().add_module(cls, module_name=mod_name, overwrite=overwrite)
    
    return decorator if _cls is None else decorator(_cls)


LossModules().add_module(SpikeRateDistributionTarget, overwrite=False)
LossModules().add_module(TargetFiringRate, overwrite=False)
LossModules().add_module(OrientationSelectivityLoss, overwrite=False)
LossModules().add_module(VoltageRegularization, overwrite=False)
LossModules().add_module(SynchronizationLoss, overwrite=False)
LossModules().add_module(EMDWeightRegularization, overwrite=False)
