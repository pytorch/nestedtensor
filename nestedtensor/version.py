__version__ = '0.0.1.dev202053116+7dff4fb'
git_version = '7dff4fb8e1af4935496fbfc3ae1e1bd8f43867fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
