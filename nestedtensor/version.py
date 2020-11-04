__version__ = '0.0.1.dev20201142+1f411e0'
git_version = '1f411e02a5a9c5881fa1edc672d6cec60bc09540'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
