__version__ = '0.1.4+b47bba4'
git_version = 'b47bba437cab2e387684db0fa16484e0f11c7f76'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
