__version__ = '0.0.1.dev2020103018+8acd4f3'
git_version = '8acd4f30345ab7a88f38417382f5538737646ac6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
