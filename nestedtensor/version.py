__version__ = '0.0.1.dev20204243+016f557'
git_version = '016f557069e56aa6b564d3913f00add66fe125f8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
