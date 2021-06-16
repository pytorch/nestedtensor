__version__ = '0.1.4+af5814e'
git_version = 'af5814e667626209ab0f885771ef78575523d5c6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
