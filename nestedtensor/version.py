__version__ = '0.1.4+0969a18'
git_version = '0969a180d76af847ff43a115e4d4f657336f51b2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
