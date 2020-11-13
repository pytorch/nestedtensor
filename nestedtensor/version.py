__version__ = '0.0.1.dev2020111322+26debb2'
git_version = '26debb2c6171e77a1ecfa5589c707569a7676c9e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
