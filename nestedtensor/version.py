__version__ = '0.1.4+753edd6'
git_version = '753edd6d1823e74de1618f2672b90336d12a39ed'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
