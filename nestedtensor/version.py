__version__ = '0.1.4+f2c96e5'
git_version = 'f2c96e5a21accf72c05ae39126df735a0878f6b9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
