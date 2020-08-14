__version__ = '0.0.1.dev202081421+854f281'
git_version = '854f2814feb3235152fb1378c59992903fe2c4a4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
