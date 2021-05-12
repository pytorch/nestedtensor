__version__ = '0.1.4+7ad51a8'
git_version = '7ad51a86bcb5c91a5e4d6114177547e1e993dcf9'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
