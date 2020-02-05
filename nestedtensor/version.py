__version__ = '0.0.1.dev2020253+0393a32'
git_version = '0393a326160a902bdeeb6a0d1b2cebf49e0c72a2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
