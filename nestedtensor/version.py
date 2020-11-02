__version__ = '0.0.1.dev202011220+e487630'
git_version = 'e487630cd9359fa88f75bc2c6d6b67fd473d286c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
