__version__ = '0.0.1.dev20204918+2bb3c06'
git_version = '2bb3c06d913db2d8b29f77b0a9aa88737444bf92'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
