__version__ = '0.0.1.dev20202144+a78c30e'
git_version = 'a78c30e0b29d34c94ed1f2453eb0665b8e0e7261'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
