__version__ = '0.1.4+9c500af'
git_version = '9c500afe3eb042340040ee8de9e6c77d1cf553d0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
