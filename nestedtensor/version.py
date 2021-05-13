__version__ = '0.1.4+2e9eee8'
git_version = '2e9eee8d291d44a269947a5a9a44c6e749604e73'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
