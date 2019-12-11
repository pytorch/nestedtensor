__version__ = '0.0.1.dev201912113+864a0ad'
git_version = '864a0ad5dee26631069b01049d82901f2951945e'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
