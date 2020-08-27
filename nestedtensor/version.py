__version__ = '0.0.1.dev20208271+e4e1d8c'
git_version = 'e4e1d8c66d44c702e17e54b9e02f62656f89f87d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
