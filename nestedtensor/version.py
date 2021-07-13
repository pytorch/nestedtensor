__version__ = '0.1.4+33fb247'
git_version = '33fb2477c856f8185f1e9c1e9a6ca28065e43cf9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
