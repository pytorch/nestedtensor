__version__ = '0.1.4+16d5ac1'
git_version = '16d5ac18d0835b4d76b3a719b8a4547d38971886'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
