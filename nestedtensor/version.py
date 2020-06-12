__version__ = '0.0.1.dev20206123+a09f6fa'
git_version = 'a09f6fa69ed660684bbfd06aa4eae08f2c61a6b0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
