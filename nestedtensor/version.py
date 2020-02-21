__version__ = '0.0.1.dev202022119+9688ece'
git_version = '9688ece01c5f2b2a50164129b7140cb51c5eeae7'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
