__version__ = '0.1.4+37b3a58'
git_version = '37b3a583e2804dadb799101c69148004e2c2ea0b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
