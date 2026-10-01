import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from bmtk.simulator.dpointnet.input_modules import InputsGeneratorMod


class DataIterator:

    def __init__(
        self,
        input_mods,
        batch_size,
        seq_len,
        ordered_populations=None,
        fetch_in_graph=False,
        strategy=None,
        prefetch_device_inputs=True,
        recover_input_errors=False,
        use_device_generation=True,
    ):
        self.input_mods = input_mods
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.ordered_populations = ordered_populations
        self.fetch_in_graph = fetch_in_graph
        self.strategy = strategy
        self.prefetch_device_inputs = prefetch_device_inputs
        self.recover_input_errors = recover_input_errors
        self.use_device_generation = use_device_generation
        self._device_generation = False
        self._prefetch_executor = None
        self._prefetch_future = None

        self.data_itrs = []
        self._is_built = False
        self._ret_list = False

    def close(self):
        if self._prefetch_future is not None:
            self._prefetch_future.cancel()
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
        groups = self.data_itrs if self._ret_list else [self.data_itrs]
        for group in groups:
            for iterator in group:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        self._prefetch_future = None
        self._prefetch_executor = None
        self._device_generation = False
        self.data_itrs = []
        self._is_built = False

    def build(self):
        self.close()
        if isinstance(self.input_mods, InputsGeneratorMod):
            self.input_mods = [self.input_mods]

        if isinstance(self.input_mods[0], (list, tuple)):
            self._build_list()
        else:
            self._build_singular()

        self._is_built = True

        if self._device_generation and self.prefetch_device_inputs:
            self._prefetch_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="dpointnet-lgn"
            )

    def _create_iterator(self, mod, seq_len, batch_size):
        if self.use_device_generation and getattr(mod, "use_device_generation", False):
            if self.fetch_in_graph or not tf.executing_eagerly():
                raise ValueError(
                    "Per-device LGN generation requires eager iterator fetching."
                )
            module_strategy = getattr(mod.rnn, "strategy", None)
            if (
                self.strategy is not None
                and module_strategy is not None
                and module_strategy is not self.strategy
            ):
                raise ValueError("All per-device inputs must share one strategy.")
            strategy = self.strategy or mod.rnn.strategy
            self.strategy = strategy
            self._device_generation = True
            return mod.create_batch_iterator(seq_len, batch_size, strategy)
        recoverable_iterator = getattr(mod, "create_recoverable_iterator", None)
        if (
            self.recover_input_errors
            and recoverable_iterator is not None
            and not self.fetch_in_graph
            and tf.executing_eagerly()
        ):
            return recoverable_iterator(seq_len, batch_size)
        generator = mod.create_generator(seq_len=seq_len)
        return (
            None
            if generator is None
            else iter(generator.batch(batch_size).prefetch(tf.data.AUTOTUNE))
        )

    def _build_singular(self):
        self._ret_list = False
        self.data_itrs = []
        if self.ordered_populations is not None:
            input_pops_order = {
                pop_name: idx for idx, pop_name in enumerate(self.ordered_populations)
            }
            ordered_mod_list = [None for _ in range(len(self.input_mods))]
            # Non-destructive reorder (no pop) so build() can be called repeatedly,
            # e.g. when the training loop rebuilds the iterator to recover from a
            # transient tf.data error.
            for _mod in self.input_mods:
                ordered_mod_list[input_pops_order[_mod.population_name]] = _mod
            self.input_mods = ordered_mod_list

        for mod in self.input_mods:
            self.data_itrs.append(
                self._create_iterator(mod, self.seq_len, self.batch_size)
            )

        self._is_built = True

    def _build_list(self):
        self._ret_list = True

        return_size = len(self.input_mods)
        ordered_populations = (
            [self.ordered_populations] * return_size
            if self.ordered_populations is None
            else [self.ordered_populations] * return_size
        )
        batch_sizes = (
            self.batch_size
            if isinstance(self.batch_size, (list, tuple))
            else [self.batch_size] * return_size
        )
        seq_lens = (
            self.seq_len
            if isinstance(self.seq_len, (list, tuple))
            else [self.seq_len] * return_size
        )

        self.data_itrs = []
        for imods, op, bs, sl in zip(
            self.input_mods, ordered_populations, batch_sizes, seq_lens
        ):
            _itrs = []
            if op is not None:
                input_pops_order = {
                    pop_name: idx
                    for idx, pop_name in enumerate(self.ordered_populations)
                }
                ordered_mod_list = [None for _ in range(len(imods))]
                # Non-destructive reorder (no pop) so build() can be called repeatedly
                # (iterator rebuild on transient tf.data error must not consume input_mods).
                for _mod in imods:
                    ordered_mod_list[input_pops_order[_mod.population_name]] = _mod
                imods = ordered_mod_list

            for mod in imods:
                _itrs.append(self._create_iterator(mod, sl, bs))
            self.data_itrs.append(_itrs)

    def next_spikes(self):
        if not self._is_built:
            self.build()

        if self._device_generation and not tf.executing_eagerly():
            raise ValueError(
                "Per-device LGN generation requires eager iterator fetching."
            )

        if self._prefetch_executor is not None:
            fetch = self._next_spikes_list if self._ret_list else self._next_spikes
            if self._prefetch_future is None:
                self._prefetch_future = self._prefetch_executor.submit(fetch)
            result = self._prefetch_future.result()
            self._prefetch_future = self._prefetch_executor.submit(fetch)
            return result

        if self.fetch_in_graph:
            if self._ret_list:
                return self._next_spikes_list_graph()
            return self._next_spikes_graph()

        if self._ret_list:
            return self._next_spikes_list()
        else:
            return self._next_spikes()

    @tf.function(reduce_retracing=True)
    def _next_spikes_list_graph(self):
        return self._next_spikes_list()

    @tf.function(reduce_retracing=True)
    def _next_spikes_graph(self):
        return self._next_spikes()

    def _next_spikes_list(self):
        spikes = []
        ys = []
        for itrs in self.data_itrs:
            _cspikes = []
            _cys = []
            for gen in itrs:
                if gen is None:
                    continue
                s, y = next(gen)
                _cspikes.append(s)
                _cys.append(y)

            concat_spikes = self._concat_spikes(_cspikes, self.strategy)
            spikes.append(concat_spikes)
            ys.append(_cys)

        return spikes, ys

    def _next_spikes(self):
        spikes = []
        ys = []
        for gen in self.data_itrs:
            if gen is None:
                continue
            s, y = next(gen)
            spikes.append(s)
            ys.append(y)

        concat_spikes = self._concat_spikes(spikes, self.strategy)
        return concat_spikes, ys

    @staticmethod
    def _concat_spikes(spikes, strategy=None):
        if not spikes:
            raise ValueError(
                "DataIterator did not receive any active spike generators."
            )

        if strategy is not None or any(
            isinstance(spike, tf.distribute.DistributedValues) for spike in spikes
        ):
            if strategy is None:
                raise ValueError(
                    "Distributed inputs require their distribution strategy."
                )

            def concatenate(context):
                replica = context.replica_id_in_sync_group
                try:
                    device = strategy.extended.worker_devices[replica]
                except RuntimeError:
                    device = None
                with tf.device(device):
                    local = [
                        (
                            strategy.experimental_local_results(spike)[replica]
                            if isinstance(spike, tf.distribute.DistributedValues)
                            else spike
                        )
                        for spike in spikes
                    ]
                    return DataIterator._concat_spikes(local)

            return strategy.experimental_distribute_values_from_function(concatenate)

        non_bool_dtypes = [spike.dtype for spike in spikes if spike.dtype != tf.bool]
        if non_bool_dtypes:
            concat_dtype = non_bool_dtypes[0]
            spikes = [
                tf.cast(spike, concat_dtype) if spike.dtype != concat_dtype else spike
                for spike in spikes
            ]

        return tf.concat(spikes, axis=2)
