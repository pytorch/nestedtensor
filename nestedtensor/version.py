__version__ = '0.0.1.dev202051520+c9b9199'
git_version = 'c9b9199cdd478917a1209f2d87629f39564c4ff9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
