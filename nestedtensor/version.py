__version__ = '0.0.1.dev202051418+d8cd7e8'
git_version = 'd8cd7e82aed3403d6d9e0d0607ef604f3690a630'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
