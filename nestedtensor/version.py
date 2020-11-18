__version__ = '0.0.1.dev202011182+e7b5b45'
git_version = 'e7b5b45c4da3400a031c00f37b7c46df2da343ee'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
