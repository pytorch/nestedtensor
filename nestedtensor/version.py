__version__ = '0.1.4+cf6535e'
git_version = 'cf6535e005d2a048f87855b3364fb07b653b3999'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
