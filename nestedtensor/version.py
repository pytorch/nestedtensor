__version__ = '0.0.1.dev20204111+5dc822c'
git_version = '5dc822cc1f11669ef0b3c96c98923be365c644f2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
