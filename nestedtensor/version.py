__version__ = '0.0.1.dev202011174+e0951e3'
git_version = 'e0951e334a763284ba79d9c2e4a31d185e70e909'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
