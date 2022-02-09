__version__ = '0.1.4+8d242cb'
git_version = '8d242cb43bfbe31ef5cff11aa116c10e451aac82'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
