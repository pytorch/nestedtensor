__version__ = '0.0.1.dev202022118+343aa82'
git_version = '343aa82d258198698e703e7e32ac651628abd132'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
