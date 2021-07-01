__version__ = '0.1.4+a6b019c'
git_version = 'a6b019c11c12ebd1b944305ccd51c5e1cb0e5262'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
