__version__ = '0.0.1+e93e897'
git_version = 'e93e8972851edac8642f7800404c497da09c3227'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
