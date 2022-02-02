__version__ = '0.1.4+b498b56'
git_version = 'b498b56cfbbdf81ff43041fb923f0b85e8381b26'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
