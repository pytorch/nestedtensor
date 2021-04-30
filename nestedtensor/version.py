__version__ = '0.1.4+1d2cf88'
git_version = '1d2cf888f3b9dc808a90dee18197e9f7b5eef19c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
