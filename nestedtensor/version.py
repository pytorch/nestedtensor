__version__ = '0.0.1.dev202051321+190d97c'
git_version = '190d97c753a00797fa7c957a5936cd906c5a0229'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
