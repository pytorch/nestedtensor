__version__ = '0.0.1.dev202071620+2d231ad'
git_version = '2d231ad64b3210c4926f2b3a0ca5b4eed4c2b21c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
