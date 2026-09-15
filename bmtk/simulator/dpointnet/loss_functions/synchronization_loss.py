import tensorflow as tf
import os
import numpy as np

from . import loss_utils

FANO_SAMPLE_SIZE_MEAN = 70
FANO_SAMPLE_SIZE_STDDEV = 30
FANO_SAMPLE_SIZE_MIN = 15
FANO_PLAN_SEED = 42


def fano_sampling_plan(n_pool, n_samples, n_trials, seed=FANO_PLAN_SEED):
    """Build the deterministic neuron and balanced-trial sampling layout."""
    if n_pool < FANO_SAMPLE_SIZE_MIN:
        raise ValueError(
            f"Fano sampling needs at least {FANO_SAMPLE_SIZE_MIN} neurons in the "
            f"pool, got {n_pool}."
        )
    if n_trials < 1 or n_samples < 1:
        raise ValueError(f"n_trials={n_trials} and n_samples={n_samples} must be >= 1.")

    per_trial = int(np.ceil(n_samples / n_trials))
    n_effective = n_trials * per_trial
    rng = np.random.default_rng(seed)
    counts = np.clip(
        rng.normal(
            FANO_SAMPLE_SIZE_MEAN,
            FANO_SAMPLE_SIZE_STDDEV,
            n_effective,
        ).astype(np.int64),
        FANO_SAMPLE_SIZE_MIN,
        n_pool,
    )

    epoch = 0
    previous_id = 0
    offsets = np.empty(n_effective, dtype=np.int64)
    for sample_index, count in enumerate(counts):
        if previous_id + count > n_pool:
            epoch += 1
            previous_id = 0
        offsets[sample_index] = epoch * n_pool + previous_id
        previous_id += count

    n_epochs = epoch + 1
    max_count = int(counts.max())
    column = np.arange(max_count, dtype=np.int64)[None, :]
    positions = offsets[:, None] + column
    neuron_mask = column < counts[:, None]
    positions = np.where(neuron_mask, positions, 0)

    return {
        "per_trial": per_trial,
        "n_effective": n_effective,
        "counts": counts,
        "offsets": offsets,
        "n_epochs": n_epochs,
        "max_count": max_count,
        "positions": positions.reshape(n_trials, per_trial * max_count).astype(
            np.int32
        ),
        "neuron_mask": neuron_mask.reshape(n_trials, per_trial, max_count).astype(
            np.float32
        ),
    }


class SynchronizationLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        rnn,
        sync_cost=10,
        t_start=0.0,
        t_end=500.0,
        n_samples=50,
        neuropixels_data_dir="Synchronization_data",
        data_dir="GLIF_network",
        session=None,
        dtype=tf.float32,
        core_mask=None,
        seed=42,
        stimulus_type="drifting_gratings",
        **kwargs,
    ):
        super(SynchronizationLoss, self).__init__(dtype=dtype)
        self._rnn = rnn
        self._network = rnn.recurrent_network
        self._sync_cost = sync_cost
        self._t_start_ms = float(t_start)
        self._t_end_ms = float(t_end)
        if self._t_end_ms <= self._t_start_ms:
            raise ValueError(
                f"SynchronizationLoss expects t_start/t_end in ms with t_end > t_start; "
                f"got t_start={t_start}, t_end={t_end}."
            )
        if 0.0 < self._t_end_ms <= 10.0:
            raise ValueError(
                "SynchronizationLoss t_start/t_end are now specified in ms. "
                f"Got t_start={t_start}, t_end={t_end}; if these are seconds, multiply by 1000 "
                f"(t_start={self._t_start_ms * 1000:g}, t_end={self._t_end_ms * 1000:g})."
            )
        self._t_start_idx = int(round(self._t_start_ms))
        self._t_end_idx = int(round(self._t_end_ms))
        duration_ms = self._t_end_idx - self._t_start_idx
        duration_s = duration_ms / 1000.0
        self._data_dir = data_dir
        # Resolve core mask from an explicit mask or a core_radius (matches reference loss_core_radius).
        self._core_mask = loss_utils.resolve_core_mask(
            self._network, core_mask, kwargs.get("core_radius"), data_dir
        )
        self._core_indices = None
        if self._core_mask is not None:
            self._core_indices = tf.constant(
                np.flatnonzero(self._core_mask), dtype=tf.int32
            )
        self._neuropixels_data_dir = neuropixels_data_dir
        self._dtype = dtype
        self._n_samples = n_samples
        max_int32 = 2**31 - 1
        self._base_seed_pair = tf.constant(
            [int(seed) % max_int32, int(seed + 15485863) % max_int32],
            dtype=tf.int32,
        )
        self._seed_stream = self.add_weight(
            name="seed_stream",
            shape=(),
            dtype=tf.int64,
            trainable=False,
            initializer="zeros",
        )
        if session is None:
            if stimulus_type in ["spontaneous", "gray"]:
                session = "spont"
            elif stimulus_type == "drifting_gratings":
                session = "evoked"
            else:
                raise ValueError(
                    f"Unknown stimulus_type: {stimulus_type}. Choose among "
                    "'spontaneous', 'gray', or 'drifting_gratings'."
                )

        pop_names = loss_utils.get_pop_names(self._network, data_dir=self._data_dir)
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        excitatory_mask = node_ei == "e"
        if self._core_mask is not None:
            core_mask = tf.get_static_value(self._core_mask)
            if core_mask is None:
                raise ValueError(
                    "SynchronizationLoss core_mask must be statically known."
                )
            core_mask = np.asarray(core_mask, dtype=bool)
            if core_mask.shape != excitatory_mask.shape:
                raise ValueError(
                    "SynchronizationLoss core_mask has shape "
                    f"{core_mask.shape}, expected {excitatory_mask.shape}."
                )
            excitatory_mask &= core_mask

        self._core_e_indices_np = np.flatnonzero(excitatory_mask).astype(np.int32)
        self._n_pool = int(self._core_e_indices_np.size)
        self._plan_seed = int(seed)
        self._plan_cache = {}

        # Pre-define bin sizes (same as experimental data)
        bin_sizes = np.logspace(-3, 0, 20)

        # using the simulation length, limit bin_sizes to define at least 2 bins
        bin_sizes_mask = bin_sizes < duration_s / 2
        bin_sizes = bin_sizes[bin_sizes_mask]
        self._bin_sizes_ms = tuple(max(1, int(round(v * 1000.0))) for v in bin_sizes)
        self._bin_sizes_ms_tf = tf.constant(self._bin_sizes_ms, dtype=tf.int32)
        self._epsilon_tf = tf.constant(1e-7, dtype=self._dtype)

        # Load the experimental data
        duration = str(duration_ms)
        experimental_data_path = os.path.join(
            self._neuropixels_data_dir,
            f"Fano_factor_v1",
            f"v1_fano_running_{duration}ms_{session}.npy",
        )

        # experimental_data_path = os.path.join(data_dir, f'all_fano_300ms_{session}.npy')
        assert os.path.exists(
            experimental_data_path
        ), f"File not found: {experimental_data_path}"
        experimental_fanos = np.load(experimental_data_path, allow_pickle=True)
        experimental_fanos_mean = np.nanmean(
            experimental_fanos[:, bin_sizes_mask], axis=0
        )
        self.experimental_fanos_mean = tf.constant(
            experimental_fanos_mean, dtype=self._dtype
        )

    def _next_seed_pair(self):
        stream_id = self._seed_stream.assign_add(tf.constant(1, dtype=tf.int64))
        seed = tf.random.experimental.stateless_fold_in(
            self._base_seed_pair, tf.cast(stream_id, tf.int32)
        )
        replica_context = tf.distribute.get_replica_context()
        replica_id = (
            tf.constant(0, dtype=tf.int32)
            if replica_context is None
            else tf.cast(replica_context.replica_id_in_sync_group, tf.int32)
        )
        return tf.random.experimental.stateless_fold_in(seed, replica_id)

    @staticmethod
    def _stateless_shuffle(values, seed):
        random_keys = tf.random.stateless_uniform(
            [tf.shape(values)[0]], seed=seed, dtype=tf.float32
        )
        return tf.gather(values, tf.argsort(random_keys, stable=True))

    def _plan(self, n_trials):
        plan = self._plan_cache.get(n_trials)
        if plan is None:
            plan = fano_sampling_plan(
                self._n_pool,
                self._n_samples,
                n_trials,
                seed=self._plan_seed,
            )
            self._plan_cache[n_trials] = plan
        return plan

    def _draw_pool(self, call_seed, n_epochs):
        values = tf.constant(self._core_e_indices_np, dtype=tf.int32)
        shuffles = []
        for epoch in range(n_epochs):
            epoch_seed = tf.random.experimental.stateless_fold_in(
                call_seed, tf.constant(3 + epoch, dtype=tf.int32)
            )
            shuffles.append(self._stateless_shuffle(values, epoch_seed))
        return shuffles[0] if n_epochs == 1 else tf.concat(shuffles, axis=0)

    @staticmethod
    def module():
        return "SynchronizationLoss"

    @tf.function(jit_compile=True)
    def pop_fano_tf(self, spikes):
        fanos = tf.TensorArray(dtype=self._dtype, size=len(self._bin_sizes_ms))
        for i, bin_size in enumerate(self._bin_sizes_ms):
            n_bins = tf.shape(spikes)[1] // bin_size
            trimmed = spikes[:, : n_bins * bin_size]
            sp_counts = tf.reduce_sum(
                tf.reshape(trimmed, [tf.shape(spikes)[0], n_bins, bin_size]),
                axis=2,
            )

            # Compute mean and variance of spike counts
            mean_count = tf.reduce_mean(sp_counts, axis=1)
            var_count = tf.math.reduce_variance(sp_counts, axis=1)
            mean_count = tf.maximum(mean_count, self._epsilon_tf)

            fano_per_sample = var_count / mean_count
            fano = tf.reduce_mean(fano_per_sample)
            fanos = fanos.write(i, fano)

        return fanos.stack()

    def __call__(self, spikes, trim=True, **kwargs):
        if self._sync_cost <= 0:
            return tf.constant(0.0, dtype=self._dtype)

        spikes = tf.convert_to_tensor(spikes)
        if spikes.shape.rank is None:
            spikes = tf.cond(
                tf.equal(tf.rank(spikes), 2),
                lambda: tf.expand_dims(spikes, axis=0),
                lambda: spikes,
            )
            spikes.set_shape([None, None, None])
        elif spikes.shape.rank == 2:
            spikes = tf.expand_dims(spikes, axis=0)

        if trim:
            spikes = spikes[:, self._t_start_idx : self._t_end_idx, :]
        duration_ms = tf.cast(tf.shape(spikes)[1], tf.int32)
        bin_limit_ms = duration_ms // 2
        bin_sizes_mask = self._bin_sizes_ms_tf < bin_limit_ms
        experimental_fanos_mean = tf.boolean_mask(
            self.experimental_fanos_mean, bin_sizes_mask
        )

        n_trials, duration = spikes.shape[0], spikes.shape[1]
        if n_trials is None or duration is None:
            raise ValueError(
                "SynchronizationLoss needs statically known batch and sequence "
                f"dimensions, got {spikes.shape}."
            )
        plan = self._plan(n_trials)
        per_trial = plan["per_trial"]
        max_count = plan["max_count"]

        call_seed = self._next_seed_pair()
        shuffled_e_ids = self._draw_pool(call_seed, plan["n_epochs"])
        sample_ids = tf.gather(
            shuffled_e_ids,
            tf.constant(plan["positions"], dtype=tf.int32),
        )

        spikes = tf.cast(spikes, self._dtype)
        gathered = tf.gather(spikes, sample_ids, axis=2, batch_dims=1)
        gathered = tf.reshape(gathered, [n_trials, duration, per_trial, max_count])
        neuron_mask = tf.constant(plan["neuron_mask"], dtype=gathered.dtype)
        selected_spikes_sample = tf.reduce_sum(
            gathered * neuron_mask[:, None, :, :], axis=3
        )
        selected_spikes_sample = tf.reshape(
            tf.transpose(selected_spikes_sample, [0, 2, 1]),
            [n_trials * per_trial, duration],
        )
        if selected_spikes_sample.dtype != self._dtype:
            selected_spikes_sample = tf.cast(selected_spikes_sample, self._dtype)

        fanos_mean = self.pop_fano_tf(selected_spikes_sample)
        fanos_mean = tf.boolean_mask(fanos_mean, bin_sizes_mask)

        # Calculate MSE between experimental and calculated Fano Factors
        mse_loss = tf.cond(
            tf.size(experimental_fanos_mean) > 0,
            lambda: tf.reduce_mean(tf.square(experimental_fanos_mean - fanos_mean)),
            lambda: tf.constant(0.0, dtype=self._dtype),
        )

        # Calculate the synchronization loss
        sync_loss = self._sync_cost * mse_loss

        return sync_loss
