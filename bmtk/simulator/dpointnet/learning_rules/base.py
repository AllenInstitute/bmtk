from dataclasses import dataclass

import tensorflow as tf

from .. import optimizers


@dataclass(frozen=True)
class WeightSurface:
    name: str
    variable: tf.Variable
    indices: tf.Tensor
    synapse_types: tf.Tensor
    source: str


@dataclass(frozen=True)
class LearningRuleObservations:
    input_spikes: tf.Tensor
    spikes: tf.Tensor
    voltages: tf.Tensor
    initial_state: tuple
    spike_learning_signal: tf.Tensor
    voltage_learning_signal: tf.Tensor
    direct_weight_gradients: tuple
    targets: object = None


class LearningRule(tf.Module):
    """Interface for DPointNet weight-update strategies."""

    uses_bptt = False
    supported_training_approaches = ("single",)

    def __init__(self, name=None):
        super().__init__(name=name)
        self.rnn = None
        self.weight_surfaces = ()

    @classmethod
    def module(cls):
        raise NotImplementedError

    def build(self, rnn):
        self.rnn = rnn
        self.weight_surfaces = tuple(self._build_weight_surfaces(rnn))

    def _build_weight_surfaces(self, rnn):
        raise NotImplementedError

    def compute_updates(self, observations):
        raise NotImplementedError

    def apply_updates(self, optimizer, updates):
        updates = tuple(updates)
        gradients = optimizers.prepare_local_gradients_for_optimizer(
            optimizer, [gradient for _, gradient in updates]
        )
        optimizer.apply_gradients(
            (gradient, surface.variable)
            for (surface, _), gradient in zip(updates, gradients)
            if gradient is not None
        )

    def on_global_step_end(self):
        pass

    def get_config(self):
        return {"name": self.module()}


class BPTTLearningRule(LearningRule):
    """Marker strategy retaining DPointNet's existing gradient path."""

    uses_bptt = True
    supported_training_approaches = (
        "single",
        "parallel",
        "series",
        "series_accumulate",
    )

    @classmethod
    def module(cls):
        return "bptt"

    def build(self, rnn):
        self.rnn = rnn

    def _build_weight_surfaces(self, rnn):
        return ()

    def compute_updates(self, observations):
        raise RuntimeError("BPTT updates are computed by TrainingEngine.")
