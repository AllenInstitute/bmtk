from .base import (
    BPTTLearningRule,
    LearningRule,
    LearningRuleObservations,
    WeightSurface,
)
from .eprop import EPropLearningRule
from .local import LocalRateHomeostasisLearningRule, PairSTDPLearningRule
from .modulated import ModulatedEligibilityLearningRule
from .neuron_local import NeuronLocalThreeFactorLearningRule
from .three_factor import ThreeFactorLearningRule


class LearningRules:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._rules = {}
        self._initialized = True

    def add_rule(self, rule_cls, name=None, overwrite=True):
        name = name or rule_cls.module()
        if name in self._rules and not overwrite:
            return
        self._rules[name] = rule_cls

    def get_rule(self, name):
        if name not in self._rules:
            available = ", ".join(sorted(self._rules))
            raise ValueError(
                f'Unknown learning rule "{name}". Available rules: {available}.'
            )
        return self._rules[name]


def add_learning_rule(rule_cls, name=None, overwrite=True):
    LearningRules().add_rule(rule_cls, name=name, overwrite=overwrite)


def register_learning_rule(_cls=None, *, name=None, overwrite=True):
    def decorator(cls):
        add_learning_rule(cls, name=name, overwrite=overwrite)
        return cls

    return decorator if _cls is None else decorator(_cls)


LearningRules().add_rule(BPTTLearningRule, overwrite=False)
LearningRules().add_rule(EPropLearningRule, overwrite=False)
LearningRules().add_rule(ThreeFactorLearningRule, overwrite=False)
LearningRules().add_rule(PairSTDPLearningRule, overwrite=False)
LearningRules().add_rule(LocalRateHomeostasisLearningRule, overwrite=False)
LearningRules().add_rule(ModulatedEligibilityLearningRule, overwrite=False)
LearningRules().add_rule(NeuronLocalThreeFactorLearningRule, overwrite=False)
