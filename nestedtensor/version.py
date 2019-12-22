__version__ = '0.0.1.dev201912226+ca4faaf'
git_version = 'ca4faafe2a96329b404bdf1cbec000eba8e16378'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
