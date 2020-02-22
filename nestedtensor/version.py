__version__ = '0.0.1.dev20202221+76a8d68'
git_version = '76a8d688cc3984ba6a666341bad4181f3ba0e97b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
