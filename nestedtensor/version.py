__version__ = '0.0.1.dev20203281+8bd3d5e'
git_version = '8bd3d5ed22e30deb7b316b8506a2f1afe75fbaad'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
