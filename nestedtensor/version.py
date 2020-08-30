__version__ = '0.0.1.dev202083018+cd6cb21'
git_version = 'cd6cb2194472c9be64b064d6bd6a8c28833f33e0'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
