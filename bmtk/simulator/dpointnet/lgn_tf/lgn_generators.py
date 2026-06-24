import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from bmtk_tf.lgn_tf.lgn import LGN


def _stateless_seed_pair(seed, salt=0):
    if seed is None:
        return None
    max_int32 = 2**31 - 1
    seed_int = int(seed) % max_int32
    salt_int = int(salt) % max_int32
    return tf.constant([seed_int, (seed_int + salt_int) % max_int32], dtype=tf.int32)


def _fold_in_seed(seed_pair, value):
    with tf.device('/CPU:0'):
        return tf.random.experimental.stateless_fold_in(
            seed_pair, tf.cast(value, tf.int32)
        )



@tf.function(jit_compile=True)
def movies_concat(movie, pre_delay, post_delay, dtype=tf.float32):
    # add an gray screen period before and after the movie
    z1 = tf.zeros((pre_delay, movie.shape[1], movie.shape[2], movie.shape[3]), dtype=dtype)
    z2 = tf.zeros((post_delay, movie.shape[1], movie.shape[2], movie.shape[3]), dtype=dtype)
    videos = tf.concat((z1, movie, z2), 0)
    return videos


@tf.function(jit_compile=True)
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
        network,
        orientation=None, 
        temporal_f=2, 
        cpd=0.04, 
        contrast=0.8,                             
        row_size=80, 
        col_size=120,
        seq_len=500, 
        pre_delay=50, 
        post_delay=50,
        current_input=False, 
        regular=False,
        bmtk_compat=True,
        return_firing_rates=False,
        rotation='ccw',  # match reference V1_GLIF_model default (flags.rotation='ccw'); cw flips drift/orientation vs the OSI-loss tuning-angle convention
        billeh_phase=False,
        dtype=tf.float32,
        seed=None):

    lgn = LGN(
        network=network,
        row_size=row_size, col_size=col_size
    )

    duration =  seq_len - pre_delay - post_delay
    base_seed = _stateless_seed_pair(seed, salt=1001)

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
            phase_seed = None
            orientation_seed = None
            spike_seed = None
            if base_seed is not None:
                sample_seed = _fold_in_seed(base_seed, sample_idx)
                orientation_seed = _fold_in_seed(sample_seed, 0)
                phase_seed = _fold_in_seed(sample_seed, 1)
                spike_seed = _fold_in_seed(sample_seed, 2)

            if orientation is None:
                # generate randdomly.
                if regular:
                    theta = (theta + 45) % 360
                else:
                    if orientation_seed is None:
                        theta = tf.random.uniform(shape=[], minval=0, maxval=360, dtype=dtype)
                    else:
                        theta = tf.random.stateless_uniform(
                            shape=[],
                            seed=orientation_seed,
                            minval=0,
                            maxval=360,
                            dtype=dtype,
                        )
            else:
                theta = orientation[sample_idx % orientation_list_len]
                # theta = orientation


            mov_theta = theta if rotation == "cw" else -theta  # flip the sign for ccw

            if billeh_phase:
                mov_theta += 180
            # Ensure theta is a Tensor to avoid tf.function retracing on Python scalars.
            mov_theta = tf.cast(mov_theta, dtype)

            # Generate a random phase
            if phase_seed is None:
                phase = tf.random.uniform(shape=[], minval=0, maxval=360, dtype=dtype)
            else:
                phase = tf.random.stateless_uniform(
                    shape=[], seed=tf.cast(phase_seed, tf.int32), minval=0, maxval=360, dtype=dtype
                )

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
            spatial = lgn.spatial_response(videos, bmtk_compat)
            del videos
            # process temporal filters and get firing rates
            firing_rates = lgn.firing_rates_from_spatial(*spatial)
            if return_firing_rates:
                # yield tf.constant(firing_rates, dtype=dtype, shape=(seq_len, n_input))
                # print('yielding')
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
                # yield _z, np.array([theta], dtype=np.float32)
            sample_idx += 1

    data_set = tf.data.Dataset.from_generator(
        _g, 
        output_signature=(
            tf.TensorSpec(shape=(seq_len, lgn.n_nodes), dtype=dtype),
            {
                'orientation': tf.TensorSpec(dtype=dtype, shape=(1,)), 
                'contrast': tf.TensorSpec(dtype=dtype, shape=(1,)), 
                'duration': tf.TensorSpec(dtype=dtype, shape=(1,))
            }
        )
    )
    return data_set


def create_grey_screen_generator(
        network, 
        row_size, 
        col_size, 
        seq_len,
        contrast_value=0.0,
        current_input=False,
        bmtk_compat=True,
        return_firing_rates=False,
        dtype=tf.float32,
        seed=None
    ):
    
    lgn = LGN(
        network=network,
        row_size=row_size, col_size=col_size
    )
    base_seed = _stateless_seed_pair(seed, salt=2001)

    def _g():
        sample_idx = 0
        while True:
            spike_seed = None
            if base_seed is not None:
                sample_seed = _fold_in_seed(base_seed, sample_idx)
                spike_seed = _fold_in_seed(sample_seed, 0)

            # Create a gray screen (all zeros)
            # gray_screen = tf.zeros((seq_len, row_size, col_size, 1), dtype=dtype)
            gray_screen = tf.ones((seq_len, row_size, col_size, 1), dtype=dtype)*contrast_value

            # Process through LGN spatial filters
            spatial = lgn.spatial_response(gray_screen, bmtk_compat)
            del gray_screen

            # Get firing rates from spatial response
            firing_rates = lgn.firing_rates_from_spatial(*spatial)

            if return_firing_rates:
                yield firing_rates, 0.0
            else:
                del spatial
                # Sample spikes from firing rates
                # Assuming dt = 1 ms
                _p = 1 - tf.exp(-firing_rates / 1000.)  # Probability of spike in dt
                del firing_rates

                if current_input:
                    _z = _p * 1.3
                else:
                    if spike_seed is None:
                        _z = tf.random.uniform(tf.shape(_p), dtype=dtype) < _p
                    else:
                        _z = tf.random.stateless_uniform(
                            tf.shape(_p), 
                            seed=spike_seed, 
                            dtype=dtype
                        ) < _p
                del _p

                yield _z, {'contrast': tf.constant(contrast_value, dtype=dtype, shape=(1,))}
            sample_idx += 1


    data_set = tf.data.Dataset.from_generator(
        _g,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, lgn.n_nodes)), 
            {
                'contrast': tf.TensorSpec(dtype=dtype, shape=(1,))
            }
        )
    )
    return data_set

