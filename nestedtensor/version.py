__version__ = '0.0.1.dev202082323+36b93f9'
git_version = '36b93f9ba04e5115044c56b24672bf0312b59fdf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
