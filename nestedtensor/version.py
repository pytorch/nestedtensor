__version__ = '0.0.1+c95a88b'
git_version = 'c95a88bbb60c8de0e59df52447d2156fb8260fa9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
