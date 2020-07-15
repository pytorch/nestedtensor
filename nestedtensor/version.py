__version__ = '0.0.1.dev20207153+4448736'
git_version = '4448736a9f135ea9939e77af5eca0ae6bbfb22bf'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
