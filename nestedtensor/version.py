__version__ = '0.0.1.dev201912170+c2bf62a'
git_version = 'c2bf62a9a17811eabe155e7ab2189e0e9931aa08'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
