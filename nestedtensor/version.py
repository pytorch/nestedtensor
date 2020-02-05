__version__ = '0.0.1.dev2020255+2d7b5cc'
git_version = '2d7b5cc5f9b3a1a3dbe5a252d9a8bc40f84c60bc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
