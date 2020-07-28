__version__ = '0.0.1.dev202072821+fe134e6'
git_version = 'fe134e6f256a2b95c922d40375c591668579e9c9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
