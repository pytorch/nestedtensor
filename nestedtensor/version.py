__version__ = '0.0.1.dev20205213+308a4a5'
git_version = '308a4a512bc1cb54ed6856a9319b444763b8466d'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
