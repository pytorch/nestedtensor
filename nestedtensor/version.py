__version__ = '0.0.1.dev202012421+be80045'
git_version = 'be8004591e9e5b16a13e5c4ccb892c620498013b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
