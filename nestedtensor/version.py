__version__ = '0.1.4+ffae971'
git_version = 'ffae971c4c887ad66ccce9f793b2e164fe3be849'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
