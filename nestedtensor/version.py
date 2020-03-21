__version__ = '0.0.1.dev20203210+acfef2a'
git_version = 'acfef2a0c536f6a6d297eec67b3dd52e698b73cb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
