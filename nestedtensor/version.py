__version__ = '0.0.1.dev202051521+6d43318'
git_version = '6d4331855c798f2c8531aacf130fbe55be47c471'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
