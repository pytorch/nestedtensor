__version__ = '0.1.4+14d7f60'
git_version = '14d7f600b7df096dbd06fde2204f0b6bb1a3e696'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
