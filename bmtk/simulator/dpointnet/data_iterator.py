import tensorflow as tf

from bmtk.simulator.dpointnet.input_modules import InputsGeneratorMod


class DataIterator:
    def __init__(self, input_mods, batch_size, seq_len, ordered_populations=None, fetch_in_graph=False):
        self.input_mods = input_mods
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.ordered_populations = ordered_populations
        self.fetch_in_graph = fetch_in_graph

        self.data_itrs = []
        self._is_built = False
        self._ret_list = False

    def close(self):
        self.data_itrs = []
        self._is_built = False

    def build(self):
        self._is_built = False
        if isinstance(self.input_mods, InputsGeneratorMod):
            self.input_mods = [self.input_mods]

        if isinstance(self.input_mods[0], (list, tuple)):
            self._build_list()
        else:
            self._build_singular()

        self._is_built = True

    def _build_singular(self):
        self._ret_list = False
        self.data_itrs = []
        if self.ordered_populations is not None:
            input_pops_order = {pop_name: idx for idx, pop_name in enumerate(self.ordered_populations)}
            ordered_mod_list = [None for _ in range(len(self.input_mods))]
            # Non-destructive reorder (no pop) so build() can be called repeatedly,
            # e.g. when the training loop rebuilds the iterator to recover from a
            # transient tf.data error.
            for _mod in self.input_mods:
                ordered_mod_list[input_pops_order[_mod.population_name]] = _mod
            self.input_mods = ordered_mod_list

        for mod in self.input_mods:
            generator = mod.create_generator(seq_len=self.seq_len)
            if generator is None:
                self.data_itrs.append(None)
            else:                    
                dataset = generator.batch(self.batch_size)
                itr = iter(dataset)
                self.data_itrs.append(itr)

        self._is_built = True

    def _build_list(self):
        self._ret_list = True

        return_size = len(self.input_mods)
        ordered_populations = [self.ordered_populations]*return_size if self.ordered_populations is None else [self.ordered_populations]*return_size
        batch_sizes = self.batch_size if isinstance(self.batch_size, (list, tuple)) else [self.batch_size]*return_size
        seq_lens = self.seq_len if isinstance(self.seq_len, (list, tuple)) else [self.seq_len]*return_size

        self.data_itrs = []
        for imods, op, bs, sl in zip(self.input_mods, ordered_populations, batch_sizes, seq_lens):
            _itrs = []
            if op is not None:
                input_pops_order = {pop_name: idx for idx, pop_name in enumerate(self.ordered_populations)}
                ordered_mod_list = [None for _ in range(len(imods))]
                # Non-destructive reorder (no pop) so build() can be called repeatedly
                # (iterator rebuild on transient tf.data error must not consume input_mods).
                for _mod in imods:
                    ordered_mod_list[input_pops_order[_mod.population_name]] = _mod
                imods = ordered_mod_list
            
            for mod in imods:
                generator = mod.create_generator(seq_len=sl)
                if generator is None:
                    _itrs.append(None)
                else:                    
                    dataset = generator.batch(bs)
                    itr = iter(dataset)
                    _itrs.append(itr)
            self.data_itrs.append(_itrs)

    def next_spikes(self):
        if not self._is_built:
            self.build()

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

            concat_spikes = self._concat_spikes(_cspikes)
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
        
        concat_spikes = self._concat_spikes(spikes)
        return concat_spikes, ys

    @staticmethod
    def _concat_spikes(spikes):
        if not spikes:
            raise ValueError('DataIterator did not receive any active spike generators.')

        non_bool_dtypes = [spike.dtype for spike in spikes if spike.dtype != tf.bool]
        if non_bool_dtypes:
            concat_dtype = non_bool_dtypes[0]
            spikes = [
                tf.cast(spike, concat_dtype) if spike.dtype != concat_dtype else spike
                for spike in spikes
            ]

        return tf.concat(spikes, axis=2)
