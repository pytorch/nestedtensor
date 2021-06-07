__version__ = '0.1.4+0c68b8c'
git_version = '0c68b8c4c9b3e5e0a1c3b9c8c471d36d7a437eb6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
