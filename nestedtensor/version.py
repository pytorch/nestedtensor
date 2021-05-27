__version__ = '0.1.4+3a8fd81'
git_version = '3a8fd81e999271b1ecdbf6cad8d1b6e1718d00c7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
