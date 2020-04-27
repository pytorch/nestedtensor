__version__ = '0.0.1.dev20204274+f461958'
git_version = 'f461958b9facf2648bf1c16f4f3a4626a695a398'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
