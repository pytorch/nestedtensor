__version__ = '0.0.1.dev20208254+c4d8dc2'
git_version = 'c4d8dc2f5786e5ddb96a7458a319c253d0261e73'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
