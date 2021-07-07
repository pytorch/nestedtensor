__version__ = '0.1.4+a990579'
git_version = 'a9905796aa87c7852e7d97cffe76fcda28ad6318'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
