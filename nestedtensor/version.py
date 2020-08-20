__version__ = '0.0.1.dev20208206+8de6dfb'
git_version = '8de6dfba9d4e61a97895ae0ec960a496e5297350'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
