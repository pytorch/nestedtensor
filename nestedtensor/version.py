__version__ = '0.0.1.dev202091422+571d845'
git_version = '571d845fa00f38d149bff877c9abfaf93ea4d96c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
