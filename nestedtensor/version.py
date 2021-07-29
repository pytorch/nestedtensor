__version__ = '0.1.4+d1a7978'
git_version = 'd1a79782d7755c9214a1c91021302c383eb80a82'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
