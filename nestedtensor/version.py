__version__ = '0.0.1.dev2020914+af1212b'
git_version = 'af1212b6fe38b47d14620dc78efeeb2a0b63a761'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
