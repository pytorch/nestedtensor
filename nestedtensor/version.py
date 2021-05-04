__version__ = '0.1.4+679287b'
git_version = '679287bc95a80537206aa8bffd80bd69687ee6d4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
