__version__ = '0.1.4+f1da348'
git_version = 'f1da348b0ea026d6647fb347616f3e446c34b34f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
