__version__ = '0.0.1.dev201912226+5cd9748'
git_version = '5cd97488b62fbf3366787b72a01a8d624380b54b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
