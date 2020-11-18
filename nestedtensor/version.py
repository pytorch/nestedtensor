__version__ = '0.0.1.dev202011183+b2fd7d2'
git_version = 'b2fd7d2fdb41b1a1a2c28b4d062715802d4ea4eb'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
