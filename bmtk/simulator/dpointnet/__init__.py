import os

os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_graph_level=0')

from .rnn_model import RNN
from .config import Config
from .input_modules import InputModules
from .loss_functions import register_loss_module, add_loss_module
from .tf_utils import cleanup_tensorflow, enable_gpu_memory_growth
