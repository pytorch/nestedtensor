__version__ = '0.0.1.dev20202172+b8345b1'
git_version = 'b8345b12c3eeec72b46cc57f4da6f12cac1c62e2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
