__version__ = '0.0.1.dev2020275+ca7d8bf'
git_version = 'ca7d8bfb4c734aa8d8f8933970840491f08767a1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
