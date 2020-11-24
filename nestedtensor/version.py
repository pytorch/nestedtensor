__version__ = '0.0.1.dev2020112422+bc8f907'
git_version = 'bc8f90719390fb9b6c74689d2d71de00a15058ce'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
