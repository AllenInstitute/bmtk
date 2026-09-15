import tensorflow as tf


def _find_target(targets, key):
    if isinstance(targets, dict):
        if key in targets:
            return targets[key]
        for value in targets.values():
            found = _find_target(value, key)
            if found is not None:
                return found
    elif isinstance(targets, (list, tuple)):
        for value in targets:
            found = _find_target(value, key)
            if found is not None:
                return found
    return None


def population_rate_logits(
    spikes,
    delays_ms,
    dt_ms,
    cue_duration_ms,
    response_window_ms,
    pool_a_indices,
    pool_b_indices,
    temperature_hz,
):
    spikes = tf.cast(spikes, tf.float32)
    delays_ms = tf.reshape(tf.cast(delays_ms, tf.float32), [-1])
    time_ms = tf.cast(tf.range(tf.shape(spikes)[1]), tf.float32) * dt_ms
    starts = cue_duration_ms + delays_ms
    response_mask = tf.logical_and(
        time_ms[tf.newaxis, :] >= starts[:, tf.newaxis],
        time_ms[tf.newaxis, :] < starts[:, tf.newaxis] + response_window_ms,
    )
    response_mask = tf.cast(response_mask[:, :, tf.newaxis], tf.float32)
    duration_seconds = response_window_ms / 1000.0
    pool_a = tf.gather(spikes, pool_a_indices, axis=2)
    pool_b = tf.gather(spikes, pool_b_indices, axis=2)
    rate_a = tf.reduce_mean(tf.reduce_sum(pool_a * response_mask, axis=1), axis=1)
    rate_b = tf.reduce_mean(tf.reduce_sum(pool_b * response_mask, axis=1), axis=1)
    return (rate_b - rate_a) / (duration_seconds * temperature_hz)


class DelayedAssociationLoss:
    """Binary cue loss from two fixed excitatory population-rate readouts."""

    def __init__(
        self,
        rnn,
        pool_a_start=0,
        pool_a_end=100,
        pool_b_start=100,
        pool_b_end=200,
        cue_duration_ms=20.0,
        response_window_ms=50.0,
        temperature_hz=5.0,
        **kwargs,
    ):
        if not (0 <= pool_a_start < pool_a_end <= pool_b_start < pool_b_end):
            raise ValueError("Readout pools must be non-empty, ordered, and disjoint.")
        if response_window_ms <= 0.0 or temperature_hz <= 0.0:
            raise ValueError("response_window_ms and temperature_hz must be positive.")
        self.dt_ms = float(rnn.dt)
        self.cue_duration_ms = float(cue_duration_ms)
        self.response_window_ms = float(response_window_ms)
        self.temperature_hz = float(temperature_hz)
        self.pool_a_indices = tf.range(pool_a_start, pool_a_end, dtype=tf.int32)
        self.pool_b_indices = tf.range(pool_b_start, pool_b_end, dtype=tf.int32)

    @staticmethod
    def module():
        return "DelayedAssociationLoss"

    def __call__(self, spikes, y, **kwargs):
        labels = _find_target(y, "class_label")
        delays = _find_target(y, "delay_ms")
        if labels is None or delays is None:
            raise ValueError(
                "DelayedAssociationLoss requires class_label and delay_ms targets."
            )
        logits = population_rate_logits(
            spikes=spikes,
            delays_ms=delays,
            dt_ms=self.dt_ms,
            cue_duration_ms=self.cue_duration_ms,
            response_window_ms=self.response_window_ms,
            pool_a_indices=self.pool_a_indices,
            pool_b_indices=self.pool_b_indices,
            temperature_hz=self.temperature_hz,
        )
        labels = tf.reshape(tf.cast(labels, tf.float32), [-1])
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )
