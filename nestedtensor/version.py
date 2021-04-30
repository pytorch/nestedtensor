__version__ = '0.1.4+7a2e782'
git_version = '7a2e782c8a81a3ffa5b35966adaa871fec8c9c5f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
