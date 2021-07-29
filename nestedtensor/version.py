__version__ = '0.1.4+2818db1'
git_version = '2818db1d83dbd475aa54aee3ab84749a4c13e911'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
