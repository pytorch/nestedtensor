__version__ = '0.0.1.dev201912264+27bc9dc'
git_version = '27bc9dc76a80351c2d97ad1e8b9d12792c1e3012'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
