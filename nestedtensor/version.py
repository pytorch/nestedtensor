__version__ = '0.1.4+a5c2dfe'
git_version = 'a5c2dfe89c11bebf7b913a6021c2bf3588658e25'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
