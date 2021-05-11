__version__ = '0.1.4+42ad927'
git_version = '42ad9271c0e950f081e61a9ab55bb825ee5f76dc'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
