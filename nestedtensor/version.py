__version__ = '0.1.4+5659e2f'
git_version = '5659e2fb2caedeac44b044151b2ebfbb5eb9a81a'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
