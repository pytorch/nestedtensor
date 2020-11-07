__version__ = '0.0.1.dev20201173+7820400'
git_version = '7820400752bd0ce4cd58849b97f4010eb3ae58cf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
