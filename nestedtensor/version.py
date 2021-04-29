__version__ = '0.1.4+5c040e0'
git_version = '5c040e037e3131e4bc56c89c2f6a4e61a65dc282'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
