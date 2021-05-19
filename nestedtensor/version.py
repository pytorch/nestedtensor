__version__ = '0.1.4+b90109e'
git_version = 'b90109e9898f6d99e37d3e3a9e301ab8b99d7e8c'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
