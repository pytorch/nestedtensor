__version__ = '0.1.4+f9e7739'
git_version = 'f9e773923c39c3ae85d84cd23f2b0d2c3f4c0775'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
