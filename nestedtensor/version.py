__version__ = '0.0.1.dev202053123+f2b83ee'
git_version = 'f2b83eeb1d4aeb36d41a704239b862d21ee53819'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
