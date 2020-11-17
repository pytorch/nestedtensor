__version__ = '0.0.1.dev202011174+5e482f9'
git_version = '5e482f9105e2a19fd486aeccf3cfe99b2d14b64e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
