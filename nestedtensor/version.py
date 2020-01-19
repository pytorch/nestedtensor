__version__ = '0.0.1.dev202011719+df1d0dd'
git_version = 'df1d0ddb6f65865288b5f2590356a6ec4f8bd5e2'
from nestedtensor import _C
if hasattr(_C, 'CUDA_VERSION'):
    cuda = _C.CUDA_VERSION
