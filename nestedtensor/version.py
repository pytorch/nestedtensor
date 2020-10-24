__version__ = '0.0.1.dev202010245+e18511f'
git_version = 'e18511f12b4f13678142ada0d66f175474bd3953'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
