__version__ = '0.1.4+0d7c7dd'
git_version = '0d7c7dda3a337ebac3555de3b3c67c996d446007'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
