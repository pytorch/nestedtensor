__version__ = '0.1.4+6a98344'
git_version = '6a98344c3b5c0a0c1422b3e6e7adcac46e7712cc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
