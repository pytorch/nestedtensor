__version__ = '0.1.4+a3bc886'
git_version = 'a3bc886abf5d219f0d65cd09d4020aaffbe35b0b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
