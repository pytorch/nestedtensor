__version__ = '0.1.4+ac1747d'
git_version = 'ac1747d2ae85b472dd9be8d6c061672edfa5c33f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
