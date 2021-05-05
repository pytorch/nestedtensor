__version__ = '0.1.4+3cddf6c'
git_version = '3cddf6c6c6d37c198112462b476ddb8e6def3bad'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
