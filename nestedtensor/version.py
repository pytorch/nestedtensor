__version__ = '0.0.1+21c3420'
git_version = '21c3420aad01db9ae70dba3612c06b6845140534'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
