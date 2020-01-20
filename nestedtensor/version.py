__version__ = '0.0.1.dev202012020+34bea94'
git_version = '34bea94f8db43aac3fc1c6754b3c2ff8485f428b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
