__version__ = '0.0.1.dev2020102721+2e9dff5'
git_version = '2e9dff5c0ae9a058a93ba9ba9500ef1be23b1ed6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
