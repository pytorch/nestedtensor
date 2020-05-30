__version__ = '0.0.1.dev20205300+ae253b3'
git_version = 'ae253b325cf0a6d7fb1f0b4acc1aa15e6d4c5f44'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
