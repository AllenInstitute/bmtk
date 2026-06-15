import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

from bmtk.simulator.dpointnet.lgn_tf.lgn import LGN
from bmtk.simulator.dpointnet.io_tools import io
from .inputs_base import InputsGeneratorMod


class LGNGenerator(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, **kwargs):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        # self.rnn = rnn
        # self.name = name       
        # self.input_network = input_network
        self.stimulus_type = kwargs['stimulus_type']
        self.stimulus_opts = kwargs['stimulus_options']
        self.row_size = self.stimulus_opts['row_size']
        self.col_size = self.stimulus_opts['col_size']
        self._cache_paths = self._resolve_cache_paths(kwargs)

        if kwargs.get('cache_overwrite', False):
            for cache_path in self._cache_paths.values():
                if cache_path is not None:
                    Path(cache_path).unlink(missing_ok=True)

        self.lgn = LGN(
            network=input_network,
            row_size=self.row_size,
            col_size=self.col_size,
            **self._cache_paths,
        )

        self._generator = None
        self._grey_screen_probabilities = None
        if self.stimulus_type == 'drifting_gratings':
            self._generator_fn = create_drifting_gratings_generator
            # self.generator_fn = create_drifting_gratings_generator(
            #     lgn_network=self.lgn,
            #     **self.stimulus_opts
            # )
        elif self.stimulus_type in ['grey_screen', 'gray_screen']:
            self._generator_fn = create_grey_screen_generator
            # self.generator_fn = create_grey_screen_generator(
            #     lgn_network=self.lgn,
            #     **self.stimulus_opts
            # )
        else:
            raise ValueError(f'{self.__module__}: Uknown stimulus_type "{self.stimulus_type}"')

        self.input_network.input_type = 'spikes'

    def create_generator(self, seq_len, dt=1.0, dtype=tf.float32):
        _seq_len = seq_len
        if _seq_len is None:
            raise ValueError(f'No "seq_len" value set, please specify number of time-steps.')
        stimulus_opts = dict(self.stimulus_opts)
        seed = stimulus_opts.pop('seed', None)
        if seed is None and self.rnn.default_seed is not None:
            if self.stimulus_type == 'drifting_gratings':
                seed = self.rnn.default_seed + 10000
            elif self.stimulus_type in ['grey_screen', 'gray_screen']:
                seed = self.rnn.default_seed + 20000
        if self.stimulus_type in ['grey_screen', 'gray_screen']:
            if self._grey_screen_probabilities is None:
                self._grey_screen_probabilities = compute_grey_screen_probabilities(
                    lgn_network=self.lgn,
                    seq_len=_seq_len,
                    dtype=dtype,
                    **stimulus_opts
                )
                self.lgn = None
            return self._generator_fn(
                probabilities=self._grey_screen_probabilities,
                seq_len=_seq_len,
                seed=seed,
                dtype=dtype,
                **stimulus_opts
            )
        return self._generator_fn(
            lgn_network=self.lgn,
            seq_len=_seq_len,
            seed=seed,
            **stimulus_opts
        )

    @staticmethod
    def module():
        return 'lgn_generator'
       
    @staticmethod
    def input_type():
        return 'spikes'

    def _resolve_cache_paths(self, kwargs):
        if not kwargs.get('cache_enabled', True):
            return {
                'spon_frs_path': None,
                'temp_krns_path': None,
                'spatial_krns_path': None,
            }

        cache_prefix = kwargs.get('cache_file', None)
        if cache_prefix:
            cache_prefix = Path(cache_prefix)
            cache_prefix.parent.mkdir(parents=True, exist_ok=True)
            cache_prefix = cache_prefix.with_suffix('')
        else:
            cache_dir = kwargs.get('cache_dir', None)
            if cache_dir:
                cache_root = Path(cache_dir)
            else:
                network_cache_file = getattr(self.input_network, '_cache_file', None)
                if network_cache_file:
                    cache_root = Path(network_cache_file).parent / 'lgn_cache'
                else:
                    cache_root = Path.cwd() / 'lgn_cache'

            cache_root.mkdir(parents=True, exist_ok=True)
            cache_key = kwargs.get('cache_key', None)
            if cache_key is None:
                n_inputs = getattr(self.input_network, 'n_nodes', 'unknown')
                cache_key = f'{self.population_name}_{n_inputs}_{self.row_size}x{self.col_size}'
            cache_prefix = cache_root / cache_key

        return {
            'spon_frs_path': str(cache_prefix.parent / f'{cache_prefix.name}.spontaneous.pkl'),
            'temp_krns_path': str(cache_prefix.parent / f'{cache_prefix.name}.temporal.pkl'),
            'spatial_krns_path': str(cache_prefix.parent / f'{cache_prefix.name}.spatial.pkl'),
        }
    

def _stateless_seed_pair(seed, salt=0):
    if seed is None:
        return None
    max_int32 = 2**31 - 1
    seed_int = int(seed) % max_int32
    salt_int = int(salt) % max_int32
    return tf.constant([seed_int, (seed_int + salt_int) % max_int32], dtype=tf.int32)


def _fold_in_seed(seed_pair, value):
    return tf.random.experimental.stateless_fold_in(
        seed_pair, tf.cast(value, tf.int32)
    )


def _sample_seed_pair(seed, salt=0, sample_idx=0, stream=0):
    seed_values = _sample_seed_values(seed, salt=salt, sample_idx=sample_idx, stream=stream)
    if seed_values is None:
        return None
    return tf.constant(seed_values, dtype=tf.int32)


def _sample_seed_values(seed, salt=0, sample_idx=0, stream=0):
    if seed is None:
        return None
    max_int32 = 2**31 - 1
    seed_int = int(seed) % max_int32
    mixed = (
        seed_int
        + int(salt) * 1009
        + int(sample_idx) * 9176
        + int(stream) * 131071
    ) % max_int32
    return mixed, (mixed + 104729) % max_int32


def _uniform_scalar(minval, maxval, seed_values=None, rng=None):
    scalar_rng = np.random.default_rng(seed_values) if seed_values is not None else rng
    if scalar_rng is None:
        scalar_rng = np.random.default_rng()
    return float(scalar_rng.uniform(minval, maxval))


def _as_scalar_tensor(value, dtype, name):
    value = tf.cast(value, dtype)
    value = tf.reshape(value, [])
    return tf.ensure_shape(value, [])


@tf.function(jit_compile=True)
def movies_concat(movie, pre_delay, post_delay, dtype=tf.float32):
    # add an gray screen period before and after the movie
    z1 = tf.zeros((pre_delay, movie.shape[1], movie.shape[2], movie.shape[3]), dtype=dtype)
    z2 = tf.zeros((post_delay, movie.shape[1], movie.shape[2], movie.shape[3]), dtype=dtype)
    videos = tf.concat((z1, movie, z2), 0)
    return videos


@tf.function(jit_compile=True) # using jit_compile can cause error with input shapes
def make_drifting_grating_stimulus(row_size=80, col_size=120, moving_flag=True, image_duration=100, cpd=0.05,
                                   temporal_f=2, theta=0, phase=0, contrast=1.0, dtype=tf.float32):
    '''
    Create the grating movie with the desired parameters
    :param t_min: start time in seconds
    :param t_max: end time in seconds
    :param cpd: cycles per degree
    :param temporal_f: in Hz
    :param theta: orientation angle
    :return: Movie object of grating with desired parameters
    '''
    #  Franz's code will accept something larger than 101 x 101 because of the
    #  kernel size.
    # row_size = row_size*2 # somehow, Franz's code only accept larger size; thus, i did the mulitplication
    # col_size = col_size*2
    frame_rate = tf.constant(1000, dtype=dtype)  # Hz
    # t_min = 0
    # t_max = tf.cast(image_duration, tf.float32) / 1000
    image_duration_f = tf.cast(image_duration, dtype=dtype)
    pi = tf.constant(np.pi, dtype=dtype)

    # assert contrast <= 1, "Contrast must be <= 1"
    # assert contrast > 0, "Contrast must be > 0"
    # tf.debugging.assert_less_equal(contrast, 1.0, message="Contrast must be <= 1")
    # tf.debugging.assert_greater(contrast, 0.0, message="Contrast must be > 0")

    # physical_spacing = 1. / (float(cpd) * 10)    #To make sure no aliasing occurs
    # 1 degree per pixel; LGN x/y are in pixel coords [0, size-1], so avoid linspace endpoint overshoot.
    # If you ever set physical_spacing != 1, you’ll need to rescale LGN x,y or change the stimulus grid size to keep alignment.
    # Otherwise, the movie shape and LGN coordinates will no longer match.
    physical_spacing = 1.0 # 1 degree, fixed for now. tf version lgn model need this to keep true cpd;
    # row_range = tf.cast(tf.linspace(0.0, row_size, tf.cast(row_size / physical_spacing, tf.int32)), dtype=dtype)
    # col_range = tf.cast(tf.linspace(0.0, col_size, tf.cast(col_size / physical_spacing, tf.int32)), dtype=dtype)
    row_range = tf.cast(tf.range(0.0, tf.cast(row_size, dtype=tf.int32), delta=int(physical_spacing)), dtype=dtype)
    col_range = tf.cast(tf.range(0.0, tf.cast(col_size, dtype=tf.int32), delta=int(physical_spacing)), dtype=dtype)
    # number_frames_needed = int(round(frame_rate * t_max))
    # number_frames_needed = tf.cast(tf.math.round(frame_rate * t_max), tf.int32)
    # time_range = tf.cast(tf.linspace(0.0, t_max, number_frames_needed), dtype=dtype)
    number_frames_needed = tf.cast(tf.math.round(image_duration_f), tf.int32)
    time_range = tf.cast(tf.range(number_frames_needed), dtype=dtype) / frame_rate

    tt, yy, xx = tf.meshgrid(time_range, row_range, col_range, indexing='ij')

    # theta_rad = tf.constant(np.pi * (180 - theta) / 180.0, dtype=dtype) #Add negative here to match brain observatory angles!
    # phase_rad = tf.constant(np.pi * (180 - phase) / 180.0, dtype=dtype)
    theta_rad = pi * (180 - theta) / 180  # Convert to radians
    phase_rad = pi * (180 - phase) / 180  # Convert to radians

    xy = xx * tf.cos(theta_rad) + yy * tf.sin(theta_rad)
    data = contrast * tf.sin(2 * pi * (cpd * xy + temporal_f * tt) + phase_rad)

    if moving_flag: # decide whether the gratings drift or they are static
        return data
    else:
        return tf.tile(data[0][tf.newaxis, ...], (image_duration, 1, 1))


def create_drifting_gratings_generator(
        lgn_network,
        seq_len,
        orientation=None, 
        temporal_f=2, 
        cpd=0.04, 
        contrast=0.8,                             
        row_size=80, 
        col_size=120,
        pre_delay=0, 
        post_delay=0,
        current_input=False, 
        regular=False,
        bmtk_compat=True,
        return_firing_rates=False,
        rotation='ccw',  # match reference V1_GLIF_model default (flags.rotation='ccw'); cw flips drift/orientation vs the OSI-loss tuning-angle convention
        billeh_phase=False,
        dtype=tf.float32,
        seed=None):

    # lgn = LGN(
    #     network=network,
    #     row_size=row_size, col_size=col_size
    # )

    duration =  seq_len - pre_delay - post_delay
    # Reference-matched stateless RNG (V1_GLIF_model/stim_dataset.py): build the base
    # stateless seed pair once, then fold in the sample index and per-stream id below.
    base_seed = None if seed is None else _stateless_seed_pair(int(seed), salt=1001)

    if orientation is not None:
        if not pd.api.types.is_list_like(orientation):
            orientation = [orientation]
        orientation_list_len = len(orientation)
    else:
        orientation_list_len = -1

    def _g():
        if regular:
            theta = -45  # to make the first one 0
        sample_idx = 0
        while True:
            orientation_seed = None
            phase_seed = None
            spike_seed = None
            if base_seed is not None:
                # Reference schedule: fold the sample index into the base pair, then fold
                # in 0/1/2 to get the orientation / phase / spike-sampling sub-streams.
                sample_seed = _fold_in_seed(base_seed, sample_idx)
                orientation_seed = _fold_in_seed(sample_seed, 0)
                phase_seed = _fold_in_seed(sample_seed, 1)
                spike_seed = _fold_in_seed(sample_seed, 2)

            if orientation is None:
                # Generate randomly, keeping theta as a Tensor to match the reference
                # stim_dataset.generate_drifting_grating_tuning path.
                if regular:
                    theta = (theta + 45) % 360
                elif orientation_seed is None:
                    theta = tf.random.uniform(shape=[], minval=0, maxval=360, dtype=dtype)
                else:
                    theta = tf.random.stateless_uniform(
                        shape=[], seed=orientation_seed, minval=0, maxval=360, dtype=dtype)
            else:
                theta = orientation[sample_idx % orientation_list_len]
                # theta = orientation


            mov_theta = theta if rotation == "cw" else -theta  # flip the sign for ccw

            if billeh_phase:
                mov_theta += 180
            # Ensure theta is a Tensor to avoid tf.function retracing on Python scalars.
            mov_theta = tf.cast(mov_theta, dtype)

            # Generate a random phase (reference-matched stateless schedule)
            if phase_seed is None:
                phase = tf.random.uniform(shape=[], minval=0, maxval=360, dtype=dtype)
            else:
                phase = tf.random.stateless_uniform(
                    shape=[], seed=phase_seed, minval=0, maxval=360, dtype=dtype)

            movie = make_drifting_grating_stimulus(
                row_size=row_size, 
                col_size=col_size, 
                moving_flag=True,
                image_duration=duration, 
                cpd=cpd, 
                temporal_f=temporal_f, 
                theta=mov_theta,
                phase=phase, 
                contrast=contrast, 
                dtype=dtype
            )

            movie = tf.expand_dims(movie, axis=-1)
            # Add an empty gray screen period before and after the movie
            videos = movies_concat(movie, pre_delay, post_delay, dtype=dtype)
            del movie
            
            # process spatial filters
            spatial = lgn_network.spatial_response(videos, bmtk_compat)
            del videos
            # process temporal filters and get firing rates
            firing_rates = lgn_network.firing_rates_from_spatial(*spatial)
            if return_firing_rates:
                # yield tf.constant(firing_rates, dtype=dtype, shape=(seq_len, n_input))
                results = firing_rates, tf.constant(theta, dtype=dtype, shape=(1,))

            else:
                del spatial
                # sample rate
                # assuming dt = 1 ms
                _p = 1 - tf.exp(-firing_rates / 1000.) # probability of having a spike before dt = 1 ms
                del firing_rates
                # _z = tf.cast(fixed_noise < _p, dtype)
                if current_input:
                    _z = _p * 1.3
                else:
                    if spike_seed is None:
                        _z = tf.random.uniform(tf.shape(_p), dtype=dtype) < _p
                    else:
                        _z = tf.random.stateless_uniform(
                            tf.shape(_p), seed=spike_seed, dtype=dtype
                        ) < _p
                del _p
                results = _z

            # yield _z, tf.constant(theta, dtype=dtype, shape=(1,)), tf.constant(contrast, dtype=dtype, shape=(1,)), tf.constant(duration, dtype=dtype, shape=(1,))
            yield results, {'orientation': tf.constant(theta, dtype=dtype, shape=(1,)), 'contrast': tf.constant(contrast, dtype=dtype, shape=(1,)), 'duration': tf.constant(duration, dtype=dtype, shape=(1,))}
            sample_idx += 1

    if return_firing_rates or current_input:
        data_dtype = dtype
    else:
        data_dtype = tf.bool

    data_set = tf.data.Dataset.from_generator(
        _g, 
        output_signature=(
            tf.TensorSpec(shape=(seq_len, lgn_network.n_nodes), dtype=data_dtype),
            {
                'orientation': tf.TensorSpec(dtype=dtype, shape=(1,)), 
                'contrast': tf.TensorSpec(dtype=dtype, shape=(1,)), 
                'duration': tf.TensorSpec(dtype=dtype, shape=(1,))
            }
        )
    )
    return data_set


def create_grey_screen_generator(
        row_size, 
        col_size, 
        seq_len,
        lgn_network=None,
        probabilities=None,
        contrast=0.0,
        current_input=False,
        bmtk_compat=True,
        return_firing_rates=False,
        dtype=tf.float32,
        seed=None
    ):
    
    # lgn = LGN(
    #     network=network,
    #     row_size=row_size, col_size=col_size
    # )
    if probabilities is None:
        probabilities = compute_grey_screen_probabilities(
            lgn_network=lgn_network,
            row_size=row_size,
            col_size=col_size,
            seq_len=seq_len,
            contrast=contrast,
            bmtk_compat=bmtk_compat,
            dtype=dtype,
        )
    probabilities = tf.convert_to_tensor(probabilities, dtype=dtype)
    # Reference-matched stateless RNG (V1_GLIF_model/stim_dataset.generate_gray_screen_stimulus).
    base_seed = None if seed is None else _stateless_seed_pair(int(seed), salt=2001)

    def _g():
        sample_idx = 0
        while True:
            spike_seed = None
            if base_seed is not None:
                # Reference schedule: fold sample index then 0 for the spike sub-stream.
                sample_seed = _fold_in_seed(base_seed, sample_idx)
                spike_seed = _fold_in_seed(sample_seed, 0)

            if return_firing_rates:
                firing_rates = -1000.0 * tf.math.log(tf.maximum(1.0 - probabilities, tf.keras.backend.epsilon()))
                yield firing_rates, {'contrast': tf.constant(contrast, dtype=dtype, shape=(1,))}
            else:
                if current_input:
                    _z = probabilities * 1.3
                else:
                    if spike_seed is None:
                        _z = tf.random.uniform(tf.shape(probabilities), dtype=dtype) < probabilities
                    else:
                        _z = tf.random.stateless_uniform(
                            tf.shape(probabilities),
                            seed=spike_seed,
                            dtype=dtype
                        ) < probabilities

                yield _z, {'contrast': tf.constant(contrast, dtype=dtype, shape=(1,))}
            sample_idx += 1


    if return_firing_rates or current_input:
        data_dtype = dtype
    else:
        data_dtype = tf.bool

    data_set = tf.data.Dataset.from_generator(
        _g,
        output_signature=(
            tf.TensorSpec(shape=probabilities.shape, dtype=data_dtype),
            {
                'contrast': tf.TensorSpec(dtype=dtype, shape=(1,))
            }
        )
        # output_signature=(
        #     tf.TensorSpec(shape=(seq_len, lgn.n_nodes)),
        #     # {
        #     #     'gs_value': tf.TensorSpec(dtype=dtype, shape=(1,))
        #     # }
        # )
    )
    return data_set


def compute_grey_screen_probabilities(
    lgn_network,
    row_size,
    col_size,
    seq_len,
    contrast=0.0,
    bmtk_compat=True,
    dtype=tf.float32,
    **kwargs,
):
    if lgn_network is None:
        raise ValueError('lgn_network is required when grey-screen probabilities are not already cached.')

    gray_screen = tf.ones((seq_len, row_size, col_size, 1), dtype=dtype)*contrast
    spatial = lgn_network.spatial_response(gray_screen, bmtk_compat)
    del gray_screen
    firing_rates = lgn_network.firing_rates_from_spatial(*spatial)
    del spatial
    probabilities = 1 - tf.exp(-firing_rates / 1000.0)
    return tf.convert_to_tensor(probabilities, dtype=dtype)
