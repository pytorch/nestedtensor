__version__ = '0.0.1.dev20201170+56dc5cb'
git_version = '56dc5cbd0d7719a285843db2a5f15c6168694bf1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
