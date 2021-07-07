__version__ = '0.1.4+d0b2717'
git_version = 'd0b2717aafbf6245c5ff6bae862aaff4dd110c90'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
