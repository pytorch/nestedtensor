__version__ = '0.0.1.dev202011420+34b1842'
git_version = '34b184282f95883e0d40f8ebafcbd27267d35ab0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
