__version__ = '0.1.4+b9179d5'
git_version = 'b9179d5912b2903339428e7290e04e14f27031a4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
