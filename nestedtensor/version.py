__version__ = '0.0.1.dev20206171+87f7c13'
git_version = '87f7c135b7b3381c279cd8d9c9c8044d0a9b2295'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
