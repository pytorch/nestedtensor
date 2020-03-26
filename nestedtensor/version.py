__version__ = '0.0.1.dev202032620+8b7ce92'
git_version = '8b7ce92a8d90a07eb3bfffac7936458ee854ac9d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
