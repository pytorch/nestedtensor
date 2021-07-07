__version__ = '0.1.4+c2af391'
git_version = 'c2af39103049bfda92ff5b51db48dfeaabc9efdd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
