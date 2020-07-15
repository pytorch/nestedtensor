__version__ = '0.0.1.dev202071520+9e06a05'
git_version = '9e06a058bf4f8f21fd28e2a04feccdcde9f7aa41'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
