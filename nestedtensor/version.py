__version__ = '0.1.4+1e0d30b'
git_version = '1e0d30b3d6811f9b8f2adcc2d5d0874c0c696178'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
