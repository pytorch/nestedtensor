__version__ = '0.0.1.dev201912720+aba3841'
git_version = 'aba3841bf6d4bb71abbf77a5b6b40927d57c7ebf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
