__version__ = '0.0.1.dev202051322+259fd05'
git_version = '259fd0551ac6677851881b7a1a6026578dc2887f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
