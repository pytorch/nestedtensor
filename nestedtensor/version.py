__version__ = '0.0.1.dev202012020+dfcac34'
git_version = 'dfcac3458dd8d539b2ee0d9dfd6ac252207148d6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
