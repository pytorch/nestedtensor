__version__ = '0.1.4+d18895e'
git_version = 'd18895e1bffae5f17dbd3449a14a5f064ddb3ca8'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
