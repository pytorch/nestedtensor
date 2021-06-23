__version__ = '0.1.4+f0e7b4b'
git_version = 'f0e7b4b2ae428ae5267c28e7263c9389aeea7dec'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
