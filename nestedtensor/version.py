__version__ = '0.1.4+8a323ab'
git_version = '8a323ab4b657843ccbf5c464b2d2ab1ad9eb7202'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
