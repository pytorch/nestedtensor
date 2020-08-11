__version__ = '0.0.1.dev202081122+9f4d266'
git_version = '9f4d2669fcf59215a1deee4fdf7fe620d73b5b4c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
