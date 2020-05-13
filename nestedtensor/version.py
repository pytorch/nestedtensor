__version__ = '0.0.1.dev20205131+3008aed'
git_version = '3008aedcee43bf332301d9d657270ee16ba31b12'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
