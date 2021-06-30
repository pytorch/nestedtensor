__version__ = '0.1.4+d80c2ce'
git_version = 'd80c2ce99ebe26c88d1195bc79baebaa6afcc223'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
