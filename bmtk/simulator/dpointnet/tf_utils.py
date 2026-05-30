import gc

import tensorflow as tf
from packaging import version

from .io_tools import io


# Necessary not to occupy all the memory on a GPU.
def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            io.log_warning(
                f'Could not set TensorFlow memory growth for {gpu.name}: {exc}'
            )


def cleanup_tensorflow():
    tf.keras.backend.clear_session()
    gc.collect()


def _get_active_policy(mixed_precision, fallback_policy=None):
    if hasattr(mixed_precision, 'global_policy'):
        return mixed_precision.global_policy()

    return fallback_policy


def get_precision_policy_and_dtype(_dtype):
    """Set TensorFlow mixed-precision policy and return the matching tf.DType. 
    Supported values: float16, bfloat16, float32, float64
    """
    dtype_name = tf.dtypes.as_dtype(_dtype).name

    if dtype_name == 'float16':
        policy_name = 'mixed_float16'
        resolved_dtype = tf.float16
        io.log_debug('Mixed precision (float16) enabled!')

    elif dtype_name == 'bfloat16':
        policy_name = 'mixed_bfloat16'
        resolved_dtype = tf.bfloat16
        io.log_debug('Mixed precision (bfloat16) enabled!')

    elif dtype_name == 'float32':
        policy_name = 'float32'
        resolved_dtype = tf.float32

    elif dtype_name == 'float64':
        policy_name = 'float64'
        resolved_dtype = tf.float64

    else:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Use one of: float16, bfloat16, float32, float64."
        )

    if version.parse(tf.__version__) < version.parse('2.4.0'):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision

        policy = mixed_precision.Policy(policy_name)
        mixed_precision.set_policy(policy)
        active_policy = _get_active_policy(mixed_precision, fallback_policy=policy)
    else:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy(policy_name)
        active_policy = _get_active_policy(mixed_precision)

    active_policy_name = getattr(active_policy, 'name', None)
    if active_policy_name != policy_name:
        raise RuntimeError(
            f'Failed to activate TensorFlow precision policy "{policy_name}". '
            f'Current policy is "{active_policy_name}".'
        )

    io.log_info(
        'TensorFlow precision policy set to '
        f'"{active_policy_name}" '
        f'(compute_dtype={active_policy.compute_dtype}, '
        f'variable_dtype={active_policy.variable_dtype}, '
        f'requested_dtype={dtype_name}).'
    )

    return mixed_precision, resolved_dtype
