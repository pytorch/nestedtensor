__version__ = '0.1.4+b02bc72'
git_version = 'b02bc7223641841853d2702532418aeba2dcc0ea'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
