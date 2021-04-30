__version__ = '0.1.4+a5eb575'
git_version = 'a5eb575481bd2bc5172d95bc0b41d155ee31beb4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
