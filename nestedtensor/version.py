__version__ = '0.1.4+3393c9f'
git_version = '3393c9f62b01ab048bde6bb63b094db27196c2d1'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
