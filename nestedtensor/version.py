__version__ = '0.0.1+b0dc454'
git_version = 'b0dc4543e0e1577330f6566cc1339747488dc235'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
