__version__ = '0.1.4+17b330a'
git_version = '17b330a5455f6e1d4b8f4a0d0c0c30bafeac5224'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
