__version__ = '0.0.1+b8ed5df'
git_version = 'b8ed5dfc728852d988e85c94af84be9b0bdcbd2a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
