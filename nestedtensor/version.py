__version__ = '0.0.1.dev20207161+8d436ed'
git_version = '8d436ed213f1b1b892b37a3b15e21f7ee1f0b8b5'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
