import numpy as np
import tensorflow as tf

from .inputs_base import InputsGeneratorMod


def generate_delayed_cue_spikes(
    rng,
    seq_len,
    n_nodes,
    dt_ms,
    label,
    delay_ms,
    cue_duration_ms,
    background_rate_hz,
    cue_rate_hz,
    probe_duration_ms=0.0,
    probe_rate_hz=0.0,
):
    cue_steps = int(round(cue_duration_ms / dt_ms))
    if cue_steps <= 0 or cue_steps > seq_len:
        raise ValueError(
            "cue_duration_ms must occupy at least one step within seq_len."
        )
    pool_size = n_nodes // 2
    if pool_size == 0:
        raise ValueError("Delayed cue input requires at least two input nodes.")

    rates = np.full((seq_len, n_nodes), background_rate_hz, dtype=np.float32)
    begin = 0 if label == 0 else pool_size
    end = pool_size if label == 0 else 2 * pool_size
    rates[:cue_steps, begin:end] += cue_rate_hz
    probe_start = int(round((cue_duration_ms + delay_ms) / dt_ms))
    probe_steps = int(round(probe_duration_ms / dt_ms))
    probe_end = min(probe_start + probe_steps, seq_len)
    if probe_steps > 0:
        rates[probe_start:probe_end, :] += probe_rate_hz
    return rng.poisson(rates * dt_ms / 1000.0).astype(np.float32)


class DelayedCueSpikes(InputsGeneratorMod):
    """Balanced two-class Poisson cues with configurable post-cue delays."""

    def __init__(
        self,
        rnn,
        name,
        input_network,
        delays_ms,
        cue_duration_ms=20.0,
        background_rate_hz=5.0,
        cue_rate_hz=80.0,
        probe_duration_ms=0.0,
        probe_rate_hz=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        input_network.input_type = "spikes"
        delays = np.asarray(delays_ms, dtype=float).reshape(-1)
        if delays.size == 0 or np.any(delays < 0.0):
            raise ValueError(
                "delays_ms must be a non-empty sequence of non-negative values."
            )
        if cue_duration_ms <= 0.0:
            raise ValueError("cue_duration_ms must be greater than zero.")
        if background_rate_hz < 0.0 or cue_rate_hz <= 0.0 or probe_rate_hz < 0.0:
            raise ValueError(
                "Poisson rates must be non-negative and cue_rate_hz positive."
            )
        if probe_duration_ms < 0.0:
            raise ValueError("probe_duration_ms must be non-negative.")
        self._delays_ms = tuple(float(delay) for delay in delays)
        self._cue_duration_ms = float(cue_duration_ms)
        self._background_rate_hz = float(background_rate_hz)
        self._cue_rate_hz = float(cue_rate_hz)
        self._probe_duration_ms = float(probe_duration_ms)
        self._probe_rate_hz = float(probe_rate_hz)
        self._seed = seed
        self._n_nodes = input_network.n_nodes

    @staticmethod
    def module():
        return "delayed_cue_spikes"

    @staticmethod
    def input_type():
        return "spikes"

    def create_generator(self, seq_len=None, dt=1.0, dtype=tf.float32, **kwargs):
        seq_len = seq_len or self.rnn.adjusted_seq_len
        dt_ms = float(getattr(self.rnn, "dt", dt))
        max_response_start = self._cue_duration_ms + max(self._delays_ms)
        if max_response_start >= seq_len * dt_ms:
            raise ValueError(
                "seq_len must extend beyond every cue-plus-delay interval."
            )
        combinations = tuple(
            (label, delay) for delay in self._delays_ms for label in (0, 1)
        )
        rng = np.random.default_rng(self._seed)
        np_dtype = tf.as_dtype(dtype).as_numpy_dtype

        def generator():
            while True:
                for index in rng.permutation(len(combinations)):
                    label, delay = combinations[index]
                    spikes = generate_delayed_cue_spikes(
                        rng=rng,
                        seq_len=seq_len,
                        n_nodes=self._n_nodes,
                        dt_ms=dt_ms,
                        label=label,
                        delay_ms=delay,
                        cue_duration_ms=self._cue_duration_ms,
                        background_rate_hz=self._background_rate_hz,
                        cue_rate_hz=self._cue_rate_hz,
                        probe_duration_ms=self._probe_duration_ms,
                        probe_rate_hz=self._probe_rate_hz,
                    ).astype(np_dtype)
                    yield spikes, {
                        "class_label": np.int32(label),
                        "delay_ms": np.float32(delay),
                    }

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(seq_len, self._n_nodes), dtype=dtype),
                {
                    "class_label": tf.TensorSpec(shape=(), dtype=tf.int32),
                    "delay_ms": tf.TensorSpec(shape=(), dtype=tf.float32),
                },
            ),
        )
