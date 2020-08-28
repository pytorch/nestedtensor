__version__ = '0.0.1.dev202082818+84e5c36'
git_version = '84e5c3610b8d6fd4aa0425fc06c27ed7fd336d4f'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
