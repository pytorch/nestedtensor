__version__ = '0.0.1.dev2020112217+f678482'
git_version = 'f6784827af87bac5d25ee268a27004b6ab051345'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
