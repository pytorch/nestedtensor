__version__ = '0.0.1.dev20206174+61a9da5'
git_version = '61a9da541a0a7659badb75c37179e7a0d7226bf8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
