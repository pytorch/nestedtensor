__version__ = '0.1.4+02ade40'
git_version = '02ade409bf604bab1c1909260eb011c43e6ce77b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
