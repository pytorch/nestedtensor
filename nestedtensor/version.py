__version__ = '0.0.1.dev202071520+01124ef'
git_version = '01124ef6f58ca89dc5c3d9d134db923ffc1bc6d6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
