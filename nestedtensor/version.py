__version__ = '0.1.4+4ed9e2f'
git_version = '4ed9e2f41e97fe1c2eda8d0d79f6edf09faad3f6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
