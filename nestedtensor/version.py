__version__ = '0.1.4+fd4af9b'
git_version = 'fd4af9bec5b8e2b8d9c443b373f395bc1e4afe91'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
