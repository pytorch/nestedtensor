__version__ = '0.0.1.dev20208516+1fa01ad'
git_version = '1fa01ad92f1a87671e74a8ecfcdec3674aa9a544'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
