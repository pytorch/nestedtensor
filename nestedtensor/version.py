__version__ = '0.0.1.dev2019122720+bce6d71'
git_version = 'bce6d71ee83b71d1f34e5bdf63498b2766deb251'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
