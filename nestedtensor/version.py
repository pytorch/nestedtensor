__version__ = '0.0.1+2c3a468'
git_version = '2c3a468de0a4a2e8c50d6dd7b41282fe98471206'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
