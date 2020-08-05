__version__ = '0.0.1.dev20208520+06bb076'
git_version = '06bb076712e55fe27790a8a4f825f15f34594c8c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
