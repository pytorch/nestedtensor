__version__ = '0.1.4+817d78b'
git_version = '817d78b40d4f4f2cb4e944eb61be93d969245138'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
