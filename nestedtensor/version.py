__version__ = '0.0.1+c559076'
git_version = 'c559076352a4af4569d167d95cc5bded106e63b3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
