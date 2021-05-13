__version__ = '0.1.4+c1e7d42'
git_version = 'c1e7d42ad5a911a4812b16d61f469e71c43c51ce'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
