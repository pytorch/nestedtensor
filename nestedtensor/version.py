__version__ = '0.0.1.dev202081918+1cafd22'
git_version = '1cafd2285f4d8f4dae1a6322e9305f3df50d2e30'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
