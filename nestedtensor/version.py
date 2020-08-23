__version__ = '0.0.1.dev202082221+9a9d176'
git_version = '9a9d1762711c45676cd182570a8068ffd511bf7f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
