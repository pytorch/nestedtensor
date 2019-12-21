__version__ = '0.0.1.dev201912211+5347bda'
git_version = '5347bdad6c72e1248069f4e4b122d7b83ea5ca95'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
