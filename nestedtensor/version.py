__version__ = '0.0.1+807ad4d'
git_version = '807ad4d72332355fd1670003b180865e8d0fcc0d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
