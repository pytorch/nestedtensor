__version__ = '0.0.1.dev202021823+d5e3e95'
git_version = 'd5e3e95510635ce9a5451b4de6e5e3df88c9a6fb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
