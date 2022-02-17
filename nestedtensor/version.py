__version__ = '0.1.4+5b45731'
git_version = '5b457313bfb6578b43d76282b321657bf85ee1b3'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
