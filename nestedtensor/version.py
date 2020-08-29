__version__ = '0.0.1.dev202082919+9d70288'
git_version = '9d7028837df97ad6a98a38fd43834e752155bd7c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
