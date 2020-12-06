__version__ = '0.0.1+e24f346'
git_version = 'e24f346c396e01b2984a3ea7e15c36bfcf73463d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
