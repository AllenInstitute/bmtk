import math

import numpy as np
import tensorflow as tf

from . import loss_utils


def low_rate_floor(spikes, neuron_ids=None, dt=1.0, floor_hz=0.1, cost=1.0):
    """Mean squared rate deficit in Hz, pooled over batch and time per neuron."""
    spikes = tf.convert_to_tensor(spikes)
    if spikes.shape.rank != 3 or spikes.shape[1] is None or spikes.shape[1] <= 0:
        raise ValueError(
            "spikes must have shape [batch, positive static time, neurons]."
        )
    rates = tf.reduce_mean(loss_utils.temporal_sum(spikes), axis=0)
    rates *= 1000.0 / (dt * spikes.shape[1])
    if neuron_ids is not None:
        rates = tf.gather(rates, neuron_ids)
    deficits = tf.nn.relu(tf.cast(floor_hz, tf.float32) - rates)
    return tf.cast(cost, tf.float32) * tf.math.divide_no_nan(
        tf.reduce_sum(tf.square(deficits)), tf.cast(tf.size(deficits), tf.float32)
    )


class LowRateFloor:
    """Opt-in one-sided firing-rate regularizer with configurable neuron selection."""

    def __init__(
        self,
        rnn,
        floor_hz=0.1,
        cost=1.0,
        neuron_ids=None,
        core_mask=None,
        core_radius=None,
        cell_types=None,
        data_dir="GLIF_network",
        **kwargs,
    ):
        self.dt = float(rnn.dt)
        self.floor_hz = float(floor_hz)
        self.cost = float(cost)
        for name, value in (
            ("dt", self.dt),
            ("floor_hz", self.floor_hz),
            ("cost", self.cost),
        ):
            if not math.isfinite(value) or value < 0 or (name == "dt" and value == 0):
                raise ValueError(
                    f"{name} must be finite and {'positive' if name == 'dt' else 'nonnegative'}."
                )
        if neuron_ids is not None and any(
            value is not None for value in (core_mask, core_radius, cell_types)
        ):
            raise ValueError("Use neuron_ids or core/cell-type selection, not both.")
        if cell_types is not None:
            cell_types = (
                [cell_types] if isinstance(cell_types, str) else list(cell_types)
            )
            unknown = set(cell_types).difference(loss_utils.CELL_TYPE_ORDER)
            if unknown:
                raise ValueError(f"Unknown cell_types: {sorted(unknown)}")
        if core_radius is not None and (
            not math.isfinite(core_radius) or core_radius <= 0
        ):
            raise ValueError("core_radius must be finite and positive.")
        if core_mask is not None or core_radius is not None or cell_types is not None:
            mask = loss_utils.resolve_core_mask(
                rnn.recurrent_network, core_mask, core_radius, data_dir
            )
            if cell_types is not None:
                populations = loss_utils.get_population_neuron_ids(
                    rnn.recurrent_network, data_dir=data_dir, core_mask=mask
                )
                neuron_ids = [
                    int(neuron)
                    for name, neurons in populations.items()
                    if name in cell_types
                    for neuron in neurons
                ]
            elif mask is not None:
                neuron_ids = np.flatnonzero(np.asarray(mask, dtype=bool))
        self.ids = None
        if neuron_ids is not None:
            ids = np.asarray(neuron_ids)
            if ids.ndim != 1 or (
                ids.size
                and (
                    not np.issubdtype(ids.dtype, np.integer)
                    or np.any(ids < 0)
                    or np.any(ids > np.iinfo(np.int32).max)
                    or np.unique(ids).size != ids.size
                )
            ):
                raise ValueError(
                    "neuron_ids must be unique nonnegative integer indices."
                )
            self.ids = tf.constant(ids.astype(np.int32), dtype=tf.int32)

    @staticmethod
    def module():
        return "LowRateFloor"

    def __call__(self, spikes, **kwargs):
        return low_rate_floor(spikes, self.ids, self.dt, self.floor_hz, self.cost)
