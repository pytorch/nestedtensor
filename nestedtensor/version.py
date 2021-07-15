__version__ = '0.1.4+e2a9d4d'
git_version = 'e2a9d4d92e0996758e6bdeeeea228b581086a928'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
