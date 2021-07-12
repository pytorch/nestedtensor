__version__ = '0.1.4+2719e68'
git_version = '2719e6833bcdec69084953381aa05e53e9df9baa'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
