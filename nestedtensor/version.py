__version__ = '0.0.1.dev20205164+11ae7e0'
git_version = '11ae7e0b6247693a16f79d6e5fe016afd540c9fd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
