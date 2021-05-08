__version__ = '0.1.4+b0aff5b'
git_version = 'b0aff5bbe2f5e5c70558cbb7dcdb9128ea1ba28a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
