__version__ = '0.0.1+0eb17b3'
git_version = '0eb17b3a3110ceb7504de0d151514b1c65c9c3f5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
USE_SUBMODULE=False
