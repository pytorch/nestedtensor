__version__ = '0.0.1.dev2020112516+b7c994f'
git_version = 'b7c994f3a16cdffafb9fc72b37746e983e570bba'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
