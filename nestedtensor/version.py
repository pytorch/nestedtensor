__version__ = '0.0.1.dev20206174+c5e37ed'
git_version = 'c5e37edebb5054a4f6638d8e203405021799bb6e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
