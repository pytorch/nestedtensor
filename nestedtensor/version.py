__version__ = '0.1.4+d6b7d8a'
git_version = 'd6b7d8a647aabcffc91bfe70336bcea5f50d7d24'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
