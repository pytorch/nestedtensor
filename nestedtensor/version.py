__version__ = '0.0.1.dev20202423+c890fe3'
git_version = 'c890fe32006c5850e1b7c937b2f883c5152254f9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
