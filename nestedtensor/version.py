__version__ = '0.0.1.dev202032015+18b2af5'
git_version = '18b2af5cf4dcb7a3a6a30202644aa56263bcbb27'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
