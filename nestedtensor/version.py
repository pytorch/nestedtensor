__version__ = '0.0.1.dev20204420+24e592a'
git_version = '24e592a426eb654a79776b4d42bd7f3ed88a8206'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
