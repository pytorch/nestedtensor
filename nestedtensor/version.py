__version__ = '0.1.4+b7db936'
git_version = 'b7db936153067adb81c698511902ffea04edcd90'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
