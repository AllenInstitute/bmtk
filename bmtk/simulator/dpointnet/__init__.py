import os

# TensorFlow reads this on first import; keep it before dpointnet modules import tf.
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')

from .rnn_model import RNN
from .config import Config
from .input_modules import InputModules
from .loss_functions import register_loss_module, add_loss_module
from .learning_rules import add_learning_rule, register_learning_rule
from .tf_utils import cleanup_tensorflow, enable_gpu_memory_growth
from .id_maps import TFIDMap


def reset():
    TFIDMap().reset()