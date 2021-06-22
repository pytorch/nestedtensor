__version__ = '0.1.4+a606464'
git_version = 'a6064649860870a014fa6bfdc814647ccf8435dd'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
