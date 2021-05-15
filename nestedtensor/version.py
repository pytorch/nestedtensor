__version__ = '0.1.4+f3fc068'
git_version = 'f3fc0689b46f5b02ba87eb22acf9102eef29ed63'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
