__version__ = '0.0.1.dev202032423+9abe210'
git_version = '9abe2108f3d5610bc49f625193670ed142da5a86'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
