__version__ = '0.0.1.dev202051618+794acee'
git_version = '794acee9f8a6d9987ea9b253ec9510b63a6f2742'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
