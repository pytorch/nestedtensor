__version__ = '0.0.1+2d7dc81'
git_version = '2d7dc817253d72054b7d2a4654eabd39a3c07e17'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
