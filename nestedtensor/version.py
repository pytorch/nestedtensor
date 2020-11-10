__version__ = '0.0.1.dev202011104+71cb387'
git_version = '71cb3879467dbb061b19726134864b4c15fdee4c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
