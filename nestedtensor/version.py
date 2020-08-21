__version__ = '0.0.1.dev202082121+b83cb4b'
git_version = 'b83cb4baf3a5ffd3a9c985cea623c665bf6d6102'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
