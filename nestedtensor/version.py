__version__ = '0.0.1.dev202082721+c92f6d2'
git_version = 'c92f6d23cc4a5c0dee8ef079ac698964445ace4b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
