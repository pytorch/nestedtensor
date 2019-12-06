__version__ = '0.0.1.dev201912619+f4a4d51'
git_version = 'f4a4d51e28d081462985362192037a7d6e5961c8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
