__version__ = '0.0.1.dev20201173+d035876'
git_version = 'd03587636729d6a995434af5bae989d35494e684'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
