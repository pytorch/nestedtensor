__version__ = '0.0.1.dev202011260+ed2a1b0'
git_version = 'ed2a1b0a2055a1f2c688d9b1418a3f072cad4713'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
