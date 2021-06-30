__version__ = '0.1.4+8329f91'
git_version = '8329f91f4ba170b8d1b145ae9c397164a1e42483'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
