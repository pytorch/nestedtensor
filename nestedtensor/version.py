__version__ = '0.0.1.dev202073120+d4ac356'
git_version = 'd4ac3565322cd339b3b8d966d3a3622d04e76bb4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
