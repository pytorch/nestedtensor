__version__ = '0.0.1.dev202011120+75e3fa6'
git_version = '75e3fa6e820aff484811f1015d0918c1a5d97831'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
