__version__ = '0.1.4+5f9f929'
git_version = '5f9f929ec1dcdd3b89625c136d2fb94102fa3238'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
