__version__ = '0.0.1.dev202082521+214d22e'
git_version = '214d22ecc723656521b30e45d6307819cf1f41c7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
