__version__ = '0.1.4+eeb96c7'
git_version = 'eeb96c783346853a0803d114faf7562e4b13e968'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
