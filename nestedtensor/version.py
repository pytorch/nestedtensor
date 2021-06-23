__version__ = '0.1.4+b2e645e'
git_version = 'b2e645e46a040814591b8206e7dda6087bfa7813'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
