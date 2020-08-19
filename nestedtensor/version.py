__version__ = '0.0.1.dev202081922+3c492ed'
git_version = '3c492ed875317e81b75b92f7d723131d865c037f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
