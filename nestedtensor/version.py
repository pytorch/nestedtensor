__version__ = '0.1.4+2c569c0'
git_version = '2c569c0eef669aaad242dc8d3b783393ee3d375e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
