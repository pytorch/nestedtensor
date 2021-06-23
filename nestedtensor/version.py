__version__ = '0.1.4+89064f2'
git_version = '89064f2b3f381d84b80917b32772e8b0bdb09181'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
