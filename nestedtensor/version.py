__version__ = '0.0.1+6a95af1'
git_version = '6a95af1cfc7efd07c7d103b3b3e5cd27148b128f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
