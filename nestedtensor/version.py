__version__ = '0.0.1.dev2020113023+ead8f43'
git_version = 'ead8f4392d24dcb3c2b5bf210bc235bff222ce40'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
