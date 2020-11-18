__version__ = '0.0.1.dev202011183+3fbbbfc'
git_version = '3fbbbfc376ea0642e46e50fa5a7683ea45bd84dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
