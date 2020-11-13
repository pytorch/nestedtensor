__version__ = '0.0.1.dev202011132+c01d5de'
git_version = 'c01d5de5bf3cce30561178253cbd4cb1eb9346dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
