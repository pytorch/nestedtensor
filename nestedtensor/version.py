__version__ = '0.0.1.dev2020661+676a1b3'
git_version = '676a1b35a97a1d836dfc49867d606aff2e92a690'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
