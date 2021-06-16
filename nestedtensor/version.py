__version__ = '0.1.4+8fe7a5b'
git_version = '8fe7a5bdf7890aba8234ee09a001829c7dd53d25'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
