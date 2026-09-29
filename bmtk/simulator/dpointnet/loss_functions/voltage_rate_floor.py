"""Replay-pure online voltage rescue with accepted-update rate history."""
import math
import inspect

import tensorflow as tf


@tf.function(jit_compile=True)
def voltage_rate_floor_step(voltage, nonrefractory, gate, target=0.9, cost=1.0):
    """FP32 neuron mean; voltage is normalized so threshold is one."""
    deficit = tf.nn.relu(tf.cast(target, tf.float32) - tf.cast(voltage, tf.float32))
    return tf.cast(cost, tf.float32) * tf.reduce_mean(
        tf.square(deficit) * tf.cast(nonrefractory, tf.float32)
        * tf.stop_gradient(tf.cast(gate, tf.float32)), axis=-1
    )


class VoltageRateFloor:
    """Opt-in all-neuron voltage floor gated by committed firing-rate EMA.

    Register before building the RNN. History pools every condition participating
    in an optimizer update, independently of orientation-loss normalizers.
    """

    def __new__(cls, rnn, cost=1.0, target=0.9, floor_hz=0.1, ema_decay=0.95,
                **kwargs):
        config = dict(cost=float(cost), target=float(target), floor_hz=float(floor_hz),
                      ema_decay=float(ema_decay), dt=float(rnn.dt))
        for loss in getattr(rnn, "_online_voltage_losses", ()):
            if isinstance(loss, cls) and loss.config == config:
                return loss
        return super().__new__(cls)

    def __init__(self, rnn, cost=1.0, target=0.9, floor_hz=0.1, ema_decay=0.95,
                 **kwargs):
        for key, value in dict(cost=cost, target=target, floor_hz=floor_hz,
                               ema_decay=ema_decay, dt=rnn.dt).items():
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")
        if cost < 0 or floor_hz <= 0 or rnn.dt <= 0 or not 0 <= ema_decay < 1:
            raise ValueError("Require cost >= 0, floor_hz/dt > 0 and 0 <= ema_decay < 1.")
        if any(key in kwargs for key in ("neuron_ids", "core_mask", "core_radius", "cell_types")):
            raise ValueError("VoltageRateFloor currently selects all model neurons.")
        if getattr(rnn, "_model_built", False):
            raise ValueError("Register VoltageRateFloor before building the RNN.")
        if hasattr(self, "config"):
            return
        self.config = dict(cost=float(cost), target=float(target), floor_hz=float(floor_hz),
                           ema_decay=float(ema_decay), dt=float(rnn.dt))
        self._pending_rates = None
        losses = getattr(rnn, "_online_voltage_losses", None)
        if losses is None:
            losses = []
            rnn._online_voltage_losses = losses
        self.index = len(losses)
        losses.append(self)

    @staticmethod
    def module():
        return "VoltageRateFloor"

    def build(self, cell):
        def weight(name, shape, value, dtype=tf.float32):
            options = {"autocast": False} if "autocast" in inspect.signature(
                cell.add_weight
            ).parameters else {"experimental_autocast": False}
            variable = cell.add_weight(
                name=f"voltage_rate_floor_{self.index}_{name}", shape=shape,
                dtype=dtype, initializer=tf.keras.initializers.Constant(value),
                trainable=False, **options
            )
            # Attribute tracking is also required by tf.train.Checkpoint on Keras 3.
            setattr(cell, f"voltage_rate_floor_{self.index}_{name}", variable)
            return variable
        for name, value in self.config.items():
            setattr(self, name, weight(name, (), value))
        self.rate_ema_hz = weight("rate_ema_hz", (cell._n_neurons,), 0.0)
        self.gate = weight("gate", (cell._n_neurons,), 0.0)
        self.initialized = weight("initialized", (), 0.0)
        self.accepted_updates = weight("accepted_updates", (), 0, tf.int64)
        if self._pending_rates is not None:
            self.initialize_rates(self._pending_rates)

    def initialize_rates(self, rates_hz):
        """Seed with measured all-neuron rates (Hz), before or after model build."""
        rates = tf.convert_to_tensor(rates_hz, tf.float32)
        tf.debugging.assert_rank(rates, 1)
        tf.debugging.assert_all_finite(rates, "rates_hz must be finite")
        tf.debugging.assert_non_negative(rates)
        if not hasattr(self, "rate_ema_hz"):
            self._pending_rates = rates
            return
        tf.debugging.assert_equal(tf.shape(rates), tf.shape(self.rate_ema_hz))
        self.rate_ema_hz.assign(rates)
        self.initialized.assign(1.0)
        self.gate.assign(tf.clip_by_value(1.0 - rates / self.floor_hz, 0.0, 1.0))

    def commit_rates(self, rates_hz):
        rates = tf.stop_gradient(tf.cast(rates_hz, tf.float32))
        updated = tf.where(
            self.initialized > 0,
            self.ema_decay * self.rate_ema_hz + (1.0 - self.ema_decay) * rates,
            rates,
        )
        self.initialize_rates(updated)
        self.accepted_updates.assign_add(1)
        return tf.constant(0)

    def step(self, voltage, nonrefractory):
        return voltage_rate_floor_step(
            voltage, nonrefractory, tf.convert_to_tensor(self.gate),
            tf.convert_to_tensor(self.target), tf.convert_to_tensor(self.cost)
        )

    def __call__(self, spikes, model_state, voltage_rate_floor_batch_weight=1.0, **kwargs):
        return tf.cast(voltage_rate_floor_batch_weight, tf.float32) * tf.reduce_mean(model_state[-1][:, self.index]) / tf.cast(
            tf.shape(spikes)[1], tf.float32
        )
