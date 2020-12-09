__version__ = '0.0.1+beacd39'
git_version = 'beacd39d7af1571093400b7f6c94f2a7a87899e4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
