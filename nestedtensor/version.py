__version__ = '0.1.4+1adeda5'
git_version = '1adeda5b7a3dd25c35be93de41335af585f15818'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
