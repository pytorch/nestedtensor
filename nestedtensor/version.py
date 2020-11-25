__version__ = '0.0.1.dev2020112522+4cf2220'
git_version = '4cf2220fc6496d4ae03588f2bc2f8bfb4b907df5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
