__version__ = '0.0.1+88da3c4'
git_version = '88da3c4433eeeb5eb06a7cef392cf234e231df96'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
