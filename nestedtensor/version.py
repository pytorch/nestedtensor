__version__ = '0.0.1.dev202082420+8defde4'
git_version = '8defde4597f4484964f33cca4745f3a214cc88d8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
