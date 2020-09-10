__version__ = '0.0.1.dev2020938+068e0b7'
git_version = '068e0b713014d12f7bc9938abf7f789429d1dd59'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
