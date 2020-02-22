__version__ = '0.0.1.dev20202222+4ba390e'
git_version = '4ba390e4d4e6af8a6f519c053e9ffd5426bf1d78'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
