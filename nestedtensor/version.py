__version__ = '0.0.1.dev202082620+35aa544'
git_version = '35aa544b7149d9e9821466e4474c3da3b876c0c9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
