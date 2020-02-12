__version__ = '0.0.1.dev202021215+4540c42'
git_version = '4540c42c1f1d2af367d4e2b375b03f3f73b2dba7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
