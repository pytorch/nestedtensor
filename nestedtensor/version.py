__version__ = '0.0.1.dev202072817+e4f6cbd'
git_version = 'e4f6cbd89f345711a0841ceded1c0f71eb748006'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
