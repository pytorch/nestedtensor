__version__ = '0.0.1.dev201912722+cf143c6'
git_version = 'cf143c6cf9e6e66cdba496ef5806378c879ea78b'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
