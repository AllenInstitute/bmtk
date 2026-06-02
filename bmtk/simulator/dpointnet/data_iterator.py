import tensorflow as tf

from bmtk.simulator.dpointnet.input_modules import InputsGeneratorMod


class DataIterator:
    def __init__(self, input_mods, batch_size, seq_len, ordered_populations=None):
        self.input_mods = input_mods
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.ordered_populations = ordered_populations

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
        if self.ordered_populations is not None:
            input_pops_order = {pop_name: idx for idx, pop_name in enumerate(self.ordered_populations)}
            ordered_mod_list = [None for _ in range(len(self.input_mods))]
            while len(self.input_mods) > 0:
                _mod = self.input_mods.pop(0)
                ordered_mod_list[input_pops_order[_mod.population_name]] = _mod
            self.input_mods = ordered_mod_list

        for mod in self.input_mods:
            generator = mod.create_generator(seq_len=self.seq_len)
            if generator is None:
                self.data_itrs.append(None)
            else:                    
                dataset = generator.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
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
                while len(imods) > 0:
                    _mod = imods.pop(0)
                    ordered_mod_list[input_pops_order[_mod.population_name]] = _mod
                imods = ordered_mod_list
            
            for mod in imods:
                generator = mod.create_generator(seq_len=sl)
                if generator is None:
                    _itrs.append(None)
                else:                    
                    dataset = generator.batch(bs).prefetch(tf.data.AUTOTUNE)
                    itr = iter(dataset)
                    _itrs.append(itr)
            self.data_itrs.append(_itrs)

    def next_spikes(self):
        if not self._is_built:
            self.build()

        if self._ret_list:
            return self._next_spikes_list()
        else:
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

            concat_spikes = tf.concat(_cspikes, axis=2)
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
        
        concat_spikes = tf.concat(spikes, axis=2)
        return concat_spikes, ys
