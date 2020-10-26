__version__ = '0.0.1.dev2020102622+63635a6'
git_version = '63635a636ef68798831cbce12500ed547953da7a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
