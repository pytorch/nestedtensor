__version__ = '0.0.1+22e7cf3'
git_version = '22e7cf372ad6cb576f6b734bec3a14c11a491d04'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
USE_SUBMODULE=False
