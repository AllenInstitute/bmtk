import tensorflow as tf
from packaging import version

from .io_tools import io


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
    else:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy(policy_name)

    return mixed_precision, resolved_dtype
