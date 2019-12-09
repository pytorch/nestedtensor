__version__ = '0.0.1.dev201912101+52f10bd'
git_version = '52f10bdfb4c48af413a72fde1b9dcab814bd85d0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
