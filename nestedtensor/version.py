__version__ = '0.0.1.dev202081822+d5515bd'
git_version = 'd5515bd7bbed7e4e6e12f9349617c0357760917a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
