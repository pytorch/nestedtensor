__version__ = '0.1.4+54a2d9e'
git_version = '54a2d9ea4feba7432a92e02a72017567f8ee99dd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
