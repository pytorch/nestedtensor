__version__ = '0.1.4+f849e47'
git_version = 'f849e476fa01489754fd3be8a966be2ead51d797'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
