__version__ = '0.0.1.dev20205193+378a25d'
git_version = '378a25dc3722d36f140f29398dec653e7dd34cb2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
