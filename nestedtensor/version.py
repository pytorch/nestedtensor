__version__ = '0.1.4+49b75a6'
git_version = '49b75a618464b11f7b25d79cf14735937d6b4bee'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
