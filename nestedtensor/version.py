__version__ = '0.1.4+048829e'
git_version = '048829e8fd8415508e6d0f0236f220244ddd52f6'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
