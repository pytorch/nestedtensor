__version__ = '0.0.1.dev202072721+ed4e944'
git_version = 'ed4e944d04d7d865ae7a8864b5c6795b9bbe540e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
