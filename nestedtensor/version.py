__version__ = '0.0.1.dev2019122620+f6e6f0c'
git_version = 'f6e6f0cae584e9f5598a7239fc68414f3560c995'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
