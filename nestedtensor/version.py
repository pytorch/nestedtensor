__version__ = '0.1.4+608b9df'
git_version = '608b9dfe433a98ea6e6edc6a0f6dc5fceb116d4e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
