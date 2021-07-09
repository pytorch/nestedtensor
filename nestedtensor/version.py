__version__ = '0.1.4+f71f9f9'
git_version = 'f71f9f9304dc9007ee737e48953d4df467739085'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
