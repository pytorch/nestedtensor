__version__ = '0.0.1.dev201912143+c6f095d'
git_version = 'c6f095ded0ec10f1cf4bc8f7eb754d0052d53a6c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
