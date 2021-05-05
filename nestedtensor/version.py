__version__ = '0.1.4+d64d157'
git_version = 'd64d1577b7b58b20d2e91520ddd4ce63101018a4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
