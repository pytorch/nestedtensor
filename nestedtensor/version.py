__version__ = '0.0.1.dev20201300+222e182'
git_version = '222e18293c6b87aad4f68d137eaeaf15957e7f59'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
