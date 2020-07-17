__version__ = '0.0.1.dev202071720+cf9f806'
git_version = 'cf9f8066c55afe5a257c125ab02bd3976d286385'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
