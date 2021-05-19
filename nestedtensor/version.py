__version__ = '0.1.4+bbef474'
git_version = 'bbef474f7ec7fe209345bc484ebd352d15e874d8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
