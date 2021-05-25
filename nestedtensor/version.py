__version__ = '0.1.4+36f9da4'
git_version = '36f9da4765e860c120388f2a280febce849f8472'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
