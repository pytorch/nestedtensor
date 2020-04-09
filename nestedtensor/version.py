__version__ = '0.0.1.dev2020490+141550b'
git_version = '141550bfad3ecf9ba0a111882800a1a5055ef632'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
