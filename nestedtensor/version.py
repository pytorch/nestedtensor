__version__ = '0.0.1.dev20201204+2a21d2d'
git_version = '2a21d2dbfc5b413742b7572f8f2fffa16bca1188'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
