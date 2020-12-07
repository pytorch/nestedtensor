__version__ = '0.0.1+be04b7d'
git_version = 'be04b7d8f681efa81d9202f07dd235210a95b87c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
