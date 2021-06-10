__version__ = '0.1.4+dc7f190'
git_version = 'dc7f1901bf38597385c4c2915088d3e8437d4781'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
