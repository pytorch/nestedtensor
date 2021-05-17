__version__ = '0.1.4+e6ee7d4'
git_version = 'e6ee7d44ab6d8539e115185d1094a97dff2df22e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
