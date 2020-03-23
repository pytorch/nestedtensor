__version__ = '0.0.1.dev202032322+c0c9486'
git_version = 'c0c94868b6be06d687a33a5147d132244f2bc415'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
