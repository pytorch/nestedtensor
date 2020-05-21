__version__ = '0.0.1.dev202052116+c98514c'
git_version = 'c98514c4c66f44d9dc2fe23dbd440e496d5981d0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
