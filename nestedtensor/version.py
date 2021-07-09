__version__ = '0.1.4+c6e09fe'
git_version = 'c6e09fed6a192fc0d51383e11d69cb3a6e62e1e4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
