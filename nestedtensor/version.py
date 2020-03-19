__version__ = '0.0.1.dev202031816+955d513'
git_version = '955d5138490c9cb16d0b3cbad6fa444a30ecb31c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
