__version__ = '0.0.1.dev202072721+78f899b'
git_version = '78f899b58377d1773a51a5e9220ca313fa4318a6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
