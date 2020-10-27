__version__ = '0.0.1.dev2020102721+d7448d6'
git_version = 'd7448d61bbaad0ad63427f31354f3f577251ad28'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
