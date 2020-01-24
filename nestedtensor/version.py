__version__ = '0.0.1.dev202012421+a7e4584'
git_version = 'a7e4584ec37298eea9adbfa67f091698563e8739'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
