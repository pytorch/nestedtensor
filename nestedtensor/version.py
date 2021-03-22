__version__ = '0.0.1+671b69e'
git_version = '671b69e868e63647d976668edb7684e915d46fe9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
