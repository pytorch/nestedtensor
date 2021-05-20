__version__ = '0.1.4+7ed5429'
git_version = '7ed5429b83cca2c9d274541b3588136af6df6bcf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
