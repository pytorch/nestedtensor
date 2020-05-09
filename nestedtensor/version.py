__version__ = '0.0.1.dev2020590+6d27a5f'
git_version = '6d27a5fe41b590d9f11a15235632721bc2eb5710'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
