__version__ = '0.0.1.dev202031922+fbca93b'
git_version = 'fbca93bae69956e674d061972dd1a3e23e6e0eea'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
