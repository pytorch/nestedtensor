__version__ = '0.1.4+f20ca2f'
git_version = 'f20ca2f38aaf234c1c5b85fc3b07fbe2e291cea5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
