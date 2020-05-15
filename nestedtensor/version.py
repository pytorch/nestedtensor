__version__ = '0.0.1.dev202051522+b38f345'
git_version = 'b38f3453a7b3a9a1376994b5b136feb687a3cca4'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
