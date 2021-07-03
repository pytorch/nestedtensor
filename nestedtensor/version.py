__version__ = '0.1.4+9da5e5c'
git_version = '9da5e5c20d110929caa77bd4a5d4e10cb9914439'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
