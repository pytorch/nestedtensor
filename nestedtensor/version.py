__version__ = '0.1.4+e3072b5'
git_version = 'e3072b508d51afe5ebadb413df7d96ea488a61f9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
