__version__ = '0.0.1+3d4432f'
git_version = '3d4432f87ec1a246176f4634934e036b97e96a61'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
USE_SUBMODULE=False
