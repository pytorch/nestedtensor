__version__ = '0.0.1.dev201912227+636e4fd'
git_version = '636e4fdd6b55c1b73c97ed3cb0455b771db4fdd9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
