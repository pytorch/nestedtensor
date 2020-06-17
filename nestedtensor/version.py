__version__ = '0.0.1.dev20206173+1426b14'
git_version = '1426b1450c4653274165b366e54d6a79db7c4f5c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
